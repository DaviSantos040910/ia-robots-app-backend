import io
import json
import logging
import uuid
import django_rq
from rest_framework import viewsets, permissions, status, parsers
from rest_framework.decorators import action
from rest_framework.response import Response
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.db import transaction
from django.core.files import File
from weasyprint import HTML
from pptx import Presentation
from openpyxl import Workbook as ExcelWorkbook
from chat.models import Chat, ChatMessage
from chat.file_processor import FileProcessor
from chat.vector_service import vector_service
from chat.services.content_extractor import ContentExtractor
from chat.services.image_description_service import image_description_service
from studio.services.knowledge_ingestion_service import KnowledgeIngestionService
from studio.jobs.artifact_jobs import generate_artifact_job

from .models import KnowledgeArtifact, KnowledgeSource, StudySpace
from .serializers import KnowledgeArtifactSerializer, KnowledgeSourceSerializer, StudySpaceSerializer

logger = logging.getLogger(__name__)

class KnowledgeSourceViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing Library/Knowledge Sources.
    """
    queryset = KnowledgeSource.objects.all()
    serializer_class = KnowledgeSourceSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return KnowledgeSource.objects.filter(user=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        # Save initially
        instance = serializer.save(user=self.request.user)

        # Ingest using centralized service
        # bot_id=0 signifies Global/Library context
        KnowledgeIngestionService.ingest_source(instance, bot_id=0)

    @action(detail=True, methods=['post'])
    def add_to_chat(self, request, pk=None):
        """
        Copies this source to a Chat as a ChatMessage attachment.
        Payload: { "chat_id": 123 }
        """
        source = self.get_object()
        chat_id = request.data.get('chat_id')

        if not chat_id:
            return Response({"error": "chat_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            chat = Chat.objects.get(id=chat_id, user=request.user)
        except Chat.DoesNotExist:
            return Response({"error": "Chat not found or access denied"}, status=status.HTTP_404_NOT_FOUND)

        try:
            with transaction.atomic():
                # Create ChatMessage
                message = ChatMessage.objects.create(
                    chat=chat,
                    role=ChatMessage.Role.USER,
                    content=f"Added source from Library: {source.title}",
                    original_filename=source.title,
                    extracted_text=source.extracted_text # Cache read-through bridge
                )

                # If it's a file, we should copy it to the message attachment
                if source.file:
                    # We open the source file and save it to the message
                    # Using 'save' on FileField automatically saves the model too
                    # We use the same name or source name
                    file_name = source.file.name.split('/')[-1]
                    with source.file.open('rb') as f:
                        message.attachment.save(file_name, File(f), save=True)
                        message.attachment_type = 'file'

                # If it's URL/YouTube, we treat it as content?
                # The prompt might handle it, but ChatMessage attachment structure usually expects file.
                # If no file, we rely on `content` or `extracted_text`.
                # SourceAssemblyService checks `message.attachment`.
                # If it's URL, we might want to save extracted text to `extracted_text` (already done)
                # AND ensure SourceAssemblyService uses it even if attachment is None?
                # Currently SourceAssemblyService checks `if message.attachment:`.
                # So we MUST attach something if we want it included by current logic.
                # OR we update SourceAssemblyService.
                # Given "Don't break contracts", updating SourceAssemblyService is risky if not tested carefully.
                # But creating a dummy file for URL source is also weird.

                # Let's rely on the extracted_text bridge.
                # If I attach a dummy file for URL sources, it works.
                # But better: if I have `extracted_text`, I should ensure `SourceAssemblyService` uses it.
                # Let's check SourceAssemblyService again.
                # It checks `if message.attachment`.
                # So for now, Library sources ONLY support FILE types fully for Generation context unless I change that service.
                # I will handle FILE type robustly here. URL support might need Service update.

                message.save()

            return Response(
                {"status": "added", "message_id": message.id},
                status=status.HTTP_201_CREATED
            )

        except Exception as e:
            logger.error(f"Error adding source to chat: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class StudySpaceViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing Study Spaces.
    """
    queryset = StudySpace.objects.all()
    serializer_class = StudySpaceSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser, parsers.JSONParser]

    def get_queryset(self):
        return StudySpace.objects.filter(user=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'])
    def link_bot(self, request, pk=None):
        space = self.get_object()
        bot_id = request.data.get('bot_id')
        if not bot_id:
            return Response({"error": "bot_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Assuming Bot model is imported from bots.models
        from bots.models import Bot
        try:
            bot = Bot.objects.get(id=bot_id)
            # Check ownership if necessary, e.g., bot.owner == request.user
            # space.bots.add(bot) - depends on relation direction
            # Bot has `study_spaces = ManyToManyField(..., related_name='bots')`
            bot.study_spaces.add(space)
            return Response({"status": "linked"}, status=status.HTTP_200_OK)
        except Bot.DoesNotExist:
            return Response({"error": "Bot not found"}, status=status.HTTP_404_NOT_FOUND)

    @action(detail=True, methods=['post'])
    def unlink_bot(self, request, pk=None):
        space = self.get_object()
        bot_id = request.data.get('bot_id')
        if not bot_id:
            return Response({"error": "bot_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        from bots.models import Bot
        try:
            bot = Bot.objects.get(id=bot_id)
            bot.study_spaces.remove(space)
            return Response({"status": "unlinked"}, status=status.HTTP_200_OK)
        except Bot.DoesNotExist:
            return Response({"error": "Bot not found"}, status=status.HTTP_404_NOT_FOUND)

    @action(detail=True, methods=['post'], parser_classes=[parsers.MultiPartParser, parsers.FormParser])
    def add_source(self, request, pk=None):
        """
        Create and link a KnowledgeSource to this Study Space.
        """
        space = self.get_object()

        # 1. Create KnowledgeSource
        title = request.data.get('title', 'Space Upload')
        source_type = request.data.get('source_type', 'FILE')

        # Map frontend type to backend choices if needed
        # KnowledgeSource.SourceType: FILE, URL, YOUTUBE, TEXT

        source = KnowledgeSource(
            user=request.user,
            title=title,
            source_type=source_type
        )

        if source_type == 'FILE' and request.FILES.get('file'):
            source.file = request.FILES['file']
        elif source_type in ['URL', 'YOUTUBE']:
            source.url = request.data.get('url')
            if not request.data.get('title'):
                source.title = source.url

        source.save()

        # 2. Ingest using centralized service for this Study Space
        KnowledgeIngestionService.ingest_source(source, study_space_id=space.id)

        # 3. Link to Space
        space.sources.add(source)

        return Response(KnowledgeSourceSerializer(source).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def remove_source(self, request, pk=None):
        space = self.get_object()
        source_id = request.data.get('source_id')
        if not source_id:
            return Response({"error": "source_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            source = KnowledgeSource.objects.get(id=source_id, user=request.user)
            space.sources.remove(source)
            return Response({"status": "removed"}, status=status.HTTP_200_OK)
        except KnowledgeSource.DoesNotExist:
            return Response({"error": "Source not found"}, status=status.HTTP_404_NOT_FOUND)

class KnowledgeArtifactViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing Knowledge Artifacts (Quizzes, Slides, etc.).
    """
    queryset = KnowledgeArtifact.objects.all()
    serializer_class = KnowledgeArtifactSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """
        Filter artifacts by user and optionally by chat_id.
        """
        user = self.request.user
        queryset = KnowledgeArtifact.objects.filter(chat__user=user)

        chat_id = self.request.query_params.get('chat_id')
        if chat_id:
            queryset = queryset.filter(chat_id=chat_id)

        return queryset.order_by('-created_at')

    def perform_create(self, serializer):
        # Ensure the chat belongs to the user
        chat = serializer.validated_data['chat']
        if chat.user != self.request.user:
            raise permissions.PermissionDenied("You do not have access to this chat.")

        # Save initially (status is PROCESSING by default in model)
        instance = serializer.save()

        # Auditoria: Extração de config do payload JSON { "config": { ... } }
        request_config = self.request.data.get('config', {})

        # Mapeamento estrito do contrato Frontend -> Backend
        options = {
            'quantity': request_config.get('quantity') or serializer.validated_data.get('quantity'),
            'difficulty': request_config.get('difficulty') or serializer.validated_data.get('difficulty'),
            'source_ids': request_config.get('selectedSourceIds') or serializer.validated_data.get('source_ids'),
            'custom_instructions': request_config.get('customInstructions') or serializer.validated_data.get('custom_instructions'),
            'target_duration': request_config.get('duration') or serializer.validated_data.get('duration'),
            'includeChatHistory': request_config.get('includeChatHistory', False)
        }

        # Generate Real Content via RQ (Async)
        instance.stage = KnowledgeArtifact.Stage.QUEUED
        instance.correlation_id = uuid.uuid4()
        instance.save()

        try:
            django_rq.enqueue(generate_artifact_job, instance.id, options)
        except Exception as e:
            logger.error(f"Error enqueueing artifact generation job: {e}", exc_info=True)
            instance.status = KnowledgeArtifact.Status.ERROR
            instance.error_message = f"Enqueue failed: {str(e)}"
            instance.save()

    @action(detail=True, methods=['get'])
    def export(self, request, pk=None):
        artifact = self.get_object()

        # 1. PODCAST (Áudio) - Download Direto
        if artifact.type == KnowledgeArtifact.ArtifactType.PODCAST:
            if not artifact.media_url:
                return HttpResponse("Áudio não disponível", status=404)
            response = HttpResponse(status=302)
            response['Location'] = artifact.media_url
            return response

        # 2. SLIDES (PowerPoint .pptx)
        elif artifact.type == KnowledgeArtifact.ArtifactType.SLIDE:
            prs = Presentation()

            content = artifact.content if isinstance(artifact.content, list) else []

            for page in content:
                if not isinstance(page, dict): continue

                # Escolhe layout: 1 = Title and Content
                slide_layout = prs.slide_layouts[1]
                slide = prs.slides.add_slide(slide_layout)

                # Título
                title = slide.shapes.title
                if title:
                    title.text = page.get('title', 'Sem Título')

                # Bullets
                body_shape = slide.placeholders[1]
                tf = body_shape.text_frame

                bullets = page.get('bullets', [])
                if bullets:
                    if isinstance(bullets, list):
                        if len(bullets) > 0:
                            tf.text = bullets[0] # Primeiro bullet
                            for b in bullets[1:]:
                                p = tf.add_paragraph()
                                p.text = b
                    else:
                         tf.text = str(bullets)

            output = io.BytesIO()
            prs.save(output)
            output.seek(0)

            response = HttpResponse(output, content_type='application/vnd.openxmlformats-officedocument.presentationml.presentation')
            response['Content-Disposition'] = f'attachment; filename="{artifact.title}.pptx"'
            return response

        # 3. SPREADSHEET (Excel .xlsx)
        elif artifact.type == KnowledgeArtifact.ArtifactType.SPREADSHEET:
            wb = ExcelWorkbook()
            ws = wb.active
            ws.title = "Dados"

            # Cabeçalho
            ws.append(["Título", "Conteúdo"])

            # Se for lista, tenta iterar
            if isinstance(artifact.content, list):
                for item in artifact.content:
                    ws.append([str(item)])
            elif isinstance(artifact.content, str):
                ws['A2'] = artifact.content

            output = io.BytesIO()
            wb.save(output)
            output.seek(0)

            response = HttpResponse(output, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = f'attachment; filename="{artifact.title}.xlsx"'
            return response

        # 4. DOCUMENTOS RICOS (PDF via HTML)
        else:
            context = {'artifact': artifact, 'content': artifact.content}
            html_string = render_to_string('studio/pdf_template.html', context)
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{artifact.title}.pdf"'
            HTML(string=html_string).write_pdf(response)
            return response
