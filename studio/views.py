import io
import json
import logging
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from django.http import HttpResponse
from django.template.loader import render_to_string
from weasyprint import HTML
from pptx import Presentation
from openpyxl import Workbook as ExcelWorkbook
from .models import KnowledgeArtifact
from .serializers import KnowledgeArtifactSerializer
from chat.services.ai_client import get_ai_client
from chat.services.context_builder import build_conversation_history
from chat.vector_service import vector_service

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

        # Extract config fields from serializer context/data
        # Note: serializer.validated_data contains write_only fields
        options = {
            'quantity': serializer.validated_data.get('quantity'),
            'difficulty': serializer.validated_data.get('difficulty'),
            'source_ids': serializer.validated_data.get('source_ids'),
            'custom_instructions': serializer.validated_data.get('custom_instructions'),
            'include_chat_history': serializer.validated_data.get('include_chat_history', False),
            'target_duration': serializer.validated_data.get('duration') # Mapped from 'duration' field
        }

        try:
            # Generate Real Content via AI
            self._generate_content_with_ai(instance, options)
        except Exception as e:
            logger.error(f"Error generating artifact content: {e}", exc_info=True)
            instance.status = KnowledgeArtifact.Status.ERROR
            instance.save()

    def _generate_content_with_ai(self, artifact, options):
        """
        Generates content using the configured AI service (Gemini/Vertex).
        """
        client = get_ai_client()
        
        # 1. Retrieve Context
        # Chat History (Conditional)
        history_text = ""
        if options.get('include_chat_history'):
            history, _ = build_conversation_history(artifact.chat_id, limit=20)
            for entry in history:
                role = entry.get('role', 'user')
                text = entry.get('parts', [{}])[0].get('text', '')
                history_text += f"{role.upper()}: {text}\n"

        # RAG Context (Sources)
        rag_context = ""
        allowed_sources = options.get('source_ids')
        
        # Determine query for RAG based on title + instructions
        # Use instructions if available, otherwise title
        query_text = options.get('custom_instructions') or f"Generar {artifact.title}"
        
        doc_contexts, _ = vector_service.search_context(
            query_text, 
            artifact.chat.user.id, 
            artifact.chat.bot.id, 
            limit=6, 
            allowed_sources=allowed_sources
        )
        
        if doc_contexts:
            rag_context = "SOURCE DOCUMENTS:\n" + "\n".join(doc_contexts) + "\n\n"

        # 2. Build Prompt based on Type
        prompt = self._build_prompt(artifact.type, artifact.title, history_text, rag_context, options)
        
        # 3. Call AI
        # Using a model that supports JSON output would be ideal.
        # Here we ask for JSON in the prompt.
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt
            )
            
            response_text = response.text
            
            # Clean up potential markdown blocks ```json ... ```
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                 response_text = response_text.split("```")[1].split("```")[0].strip()

            # 4. Parse & Save
            if artifact.type == KnowledgeArtifact.ArtifactType.PODCAST:
                # Podcast generation is complex (TTS). For MVP, we might still mock the URL 
                # or if we have a TTS service, call it here.
                # Assuming the instruction implies TEXT generation for content, 
                # but Podcast needs audio.
                # Fallback to a default audio for MVP unless we have a TTS service ready.
                artifact.media_url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
                artifact.duration = "5:30"
                # Optionally save the script
                # artifact.content = json.loads(response_text) 
            else:
                artifact.content = json.loads(response_text)

            artifact.status = KnowledgeArtifact.Status.READY
            artifact.save()

        except Exception as e:
            logger.error(f"AI Generation Failed: {e}")
            artifact.status = KnowledgeArtifact.Status.ERROR
            artifact.save()

    def _build_prompt(self, artifact_type, title, history, rag_context, options):
        difficulty = options.get('difficulty', 'Medium')
        quantity = options.get('quantity', 10)
        instructions = options.get('custom_instructions', '')
        target_duration = options.get('target_duration', 'Medium')

        base_instruction = (
            f"You are an expert educational content generator. "
            f"Create a {artifact_type} titled '{title}'.\n"
            f"Output MUST be a valid, strict JSON object. Do not include any preamble or markdown.\n"
            f"Language: Detect the language from the context (default to Portuguese if unclear).\n"
        )
        
        # Only add difficulty context if not Podcast (or logic as requested)
        # But user requested mapping Duration to word count for podcast.
        if artifact_type != KnowledgeArtifact.ArtifactType.PODCAST:
             base_instruction += f"Target Audience Difficulty: {difficulty}.\n\n"

        if instructions:
            base_instruction += f"CUSTOM INSTRUCTIONS:\n{instructions}\n\n"

        if rag_context:
            base_instruction += f"REFERENCE MATERIALS (Use these as the primary source of truth):\n{rag_context}\n"
        
        if history:
            base_instruction += f"CONVERSATION HISTORY (Context):\n{history}\n"

        base_instruction += "\nJSON STRUCTURE REQUIRED:\n"

        if artifact_type == KnowledgeArtifact.ArtifactType.QUIZ:
            return base_instruction + (
                "[\n"
                "  {\n"
                "    \"question\": \"Question text\",\n"
                "    \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
                "    \"correctAnswerIndex\": 0\n"
                "  }\n"
                "]\n"
                f"Generate exactly {quantity} questions."
            )
        
        elif artifact_type == KnowledgeArtifact.ArtifactType.SLIDE:
            return base_instruction + (
                "[\n"
                "  {\n"
                "    \"title\": \"Slide Title\",\n"
                "    \"bullets\": [\"Bullet 1\", \"Bullet 2\"]\n"
                "  }\n"
                "]\n"
                f"Generate exactly {quantity} slides."
            )

        elif artifact_type == KnowledgeArtifact.ArtifactType.FLASHCARD:
            return base_instruction + (
                "[\n"
                "  {\n"
                "    \"front\": \"Concept/Term\",\n"
                "    \"back\": \"Definition/Explanation\"\n"
                "  }\n"
                "]\n"
                f"Generate exactly {quantity} cards."
            )

        elif artifact_type == KnowledgeArtifact.ArtifactType.SPREADSHEET:
            return base_instruction + (
                 "[\n"
                 "  [\"Header 1\", \"Header 2\"],\n"
                 "  [\"Row 1 Col 1\", \"Row 1 Col 2\"]\n"
                 "]\n"
                 "Generate a dataset relevant to the topic."
            )
        
        elif artifact_type == KnowledgeArtifact.ArtifactType.PODCAST:
            # Duration Mapping
            if target_duration == 'Short':
                duration_prompt = "Create a concise 5-minute summary script (approx 800 words). Focus only on key takeaways."
            elif target_duration == 'Long':
                duration_prompt = "Create an extensive 30-minute comprehensive guide (approx 3500 words). Cover all nuances and sources in depth."
            else: # Medium
                duration_prompt = "Create a detailed 15-minute deep-dive script (approx 2000 words). Include examples and elaboration."

            return base_instruction + (
                 "{\n"
                 "  \"title\": \"Episode Title\",\n"
                 "  \"script\": \"Full text of the podcast script...\",\n"
                 "  \"hosts\": [\"Host 1\", \"Host 2\"]\n"
                 "}\n"
                 f"DURATION CONSTRAINT: {duration_prompt}"
            )

        elif artifact_type == KnowledgeArtifact.ArtifactType.SUMMARY:
             return base_instruction + (
                 "Output should be a JSON string, e.g. \"Summary text...\" "
                 "or a JSON object with a text field if preferred, but simpler is better."
             )
        
        else:
             return base_instruction + "Generate a detailed summary in JSON format."

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
                    tf.text = bullets[0] # Primeiro bullet
                    for b in bullets[1:]:
                        p = tf.add_paragraph()
                        p.text = b

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
                    # Simplificação: Dump do item como string se for complexo
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
        # Aplica-se a: WORKBOOK, FLASHCARD, QUIZ, SUMMARY
        else:
            # Prepara o contexto para o template Jinja2/Django
            context = {'artifact': artifact, 'content': artifact.content}

            # Renderiza HTML
            html_string = render_to_string('studio/pdf_template.html', context)

            # Converte para PDF
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{artifact.title}.pdf"'

            HTML(string=html_string).write_pdf(response)

            return response
