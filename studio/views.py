import io
import json
import logging
import threading
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from django.http import HttpResponse
from django.template.loader import render_to_string
from weasyprint import HTML
from pptx import Presentation
from openpyxl import Workbook as ExcelWorkbook
from google.genai import types

from .models import KnowledgeArtifact
from .serializers import KnowledgeArtifactSerializer
from chat.services.ai_client import get_ai_client, get_model
from studio.services.source_assembler import SourceAssemblyService
from studio.services.podcast_scripting import PodcastScriptingService
from studio.services.audio_mixer import AudioMixerService
from studio.schemas import QUIZ_SCHEMA, FLASHCARD_SCHEMA, SUMMARY_SCHEMA, SLIDE_SCHEMA

logger = logging.getLogger(__name__)

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
            'target_duration': request_config.get('duration') or serializer.validated_data.get('duration')
        }

        # Generate Real Content via AI (Async)
        try:
            # Pass only ID to thread to avoid stale object issues
            threading.Thread(
                target=self._generate_content_with_ai,
                args=(instance.id, options)
            ).start()
        except Exception as e:
            logger.error(f"Error starting artifact generation thread: {e}", exc_info=True)
            instance.status = KnowledgeArtifact.Status.ERROR
            instance.save()

    def _generate_content_with_ai(self, artifact_id, options):
        """
        Generates content using the configured AI service (Gemini/Vertex) with Structured Output.
        """
        try:
            # Re-fetch artifact to ensure we have the latest state and db connection in this thread
            artifact = KnowledgeArtifact.objects.get(id=artifact_id)
        except KnowledgeArtifact.DoesNotExist:
            logger.error(f"Artifact {artifact_id} not found in generation thread.")
            return

        try:
            # 1. Retrieve Context using SourceAssemblyService
            config = {
                'selectedSourceIds': options.get('source_ids', []),
                # 'includeChatContext': Removed per audit requirement
            }
            full_context = SourceAssemblyService.get_context_from_config(artifact.chat.id, config)

            # Handle Podcast flow separately
            if artifact.type == KnowledgeArtifact.ArtifactType.PODCAST:
                self._generate_podcast(artifact, full_context, options)
                return

            # Standard Artifact Generation (Quiz, Slide, etc.)
            client = get_ai_client()
            model_name = get_model('chat')

            # 2. Build Prompt
            system_instruction, response_schema = self._build_prompt_and_schema(artifact.type, artifact.title, full_context, options)

            # 3. Call AI with Structured Output
            generate_config = types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
            
            if response_schema:
                generate_config.response_schema = response_schema

            response = client.models.generate_content(
                model=model_name,
                contents="Generate the artifact content based on the system instructions and context.",
                config=generate_config
            )

            # 4. Parse & Save
            if response.parsed:
                artifact.content = response.parsed
            else:
                try:
                    artifact.content = json.loads(response.text)
                except:
                     if artifact.type == KnowledgeArtifact.ArtifactType.SUMMARY:
                         artifact.content = {"summary": response.text}
                     else:
                         raise ValueError("Failed to parse JSON response")

            artifact.status = KnowledgeArtifact.Status.READY
            artifact.save()

        except Exception as e:
            logger.error(f"AI Generation Failed for artifact {artifact_id}: {e}")
            artifact.status = KnowledgeArtifact.Status.ERROR
            artifact.save()

    def _generate_podcast(self, artifact, context, options):
        """Helper to handle podcast generation logic."""
        try:
            # 1. Generate Script
            script = PodcastScriptingService.generate_script(
                title=artifact.title,
                context=context,
                duration_constraint=options.get('target_duration', 'Medium')
            )
            artifact.content = script
            # Save script progress so user sees something if mixer fails?
            # Or keep processing.
            artifact.save()

            # 2. Mix Audio
            audio_path = AudioMixerService.mix_podcast(script)

            # Update artifact with media URL (assuming relative path matches MEDIA_URL config)
            artifact.media_url = f"/media/{audio_path}"
            artifact.duration = options.get('target_duration', '10:00')

            artifact.status = KnowledgeArtifact.Status.READY
            artifact.save()

        except Exception as e:
            logger.error(f"Podcast Generation Failed: {e}")
            artifact.status = KnowledgeArtifact.Status.ERROR
            artifact.save()

    def _build_prompt_and_schema(self, artifact_type, title, context, options):
        difficulty = options.get('difficulty', 'Medium')
        quantity = options.get('quantity', 10)
        instructions = options.get('custom_instructions', '')

        base_instruction = (
            f"You are an expert educational content generator. "
            f"Create a {artifact_type} titled '{title}'.\n"
            f"Language: Detect the language from the context (default to Portuguese if unclear).\n"
            f"Target Audience Difficulty: {difficulty}.\n"
        )
        
        if instructions:
            base_instruction += f"CUSTOM INSTRUCTIONS:\n{instructions}\n"

        base_instruction += f"\nCONTEXT MATERIAL (Source Files Only):\n{context}\n"

        schema = None

        if artifact_type == KnowledgeArtifact.ArtifactType.QUIZ:
            base_instruction += f"Generate exactly {quantity} questions."
            schema = QUIZ_SCHEMA
        
        elif artifact_type == KnowledgeArtifact.ArtifactType.FLASHCARD:
            base_instruction += f"Generate exactly {quantity} cards."
            schema = FLASHCARD_SCHEMA

        elif artifact_type == KnowledgeArtifact.ArtifactType.SUMMARY:
            base_instruction += "Generate a comprehensive summary and key points."
            schema = SUMMARY_SCHEMA

        elif artifact_type == KnowledgeArtifact.ArtifactType.SLIDE:
            base_instruction += f"Generate exactly {quantity} slides."
            schema = SLIDE_SCHEMA
        
        return base_instruction, schema

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
