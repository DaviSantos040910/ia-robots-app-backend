import io
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from django.http import HttpResponse
from django.template.loader import render_to_string
from weasyprint import HTML
from pptx import Presentation
from openpyxl import Workbook as ExcelWorkbook
from .models import KnowledgeArtifact
from .serializers import KnowledgeArtifactSerializer

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
        serializer.save()

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
