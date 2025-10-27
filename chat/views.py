# chat/views.py
from rest_framework import generics, permissions, status, parsers # Adicionar parsers
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from .models import Chat, ChatMessage
# Ajustar imports do serializer
from .serializers import (
    ChatListSerializer, ChatMessageSerializer, ChatMessageAttachmentSerializer
)
from bots.models import Bot
from django.shortcuts import get_object_or_404
from .ai_service import get_ai_response
from myproject.pagination import StandardMessagePagination
import re
import mimetypes # Para detectar o tipo de ficheiro
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework.exceptions import ValidationError as DRFValidationError


class ActiveChatListView(generics.ListAPIView):
    # ... (sem alterações) ...
    serializer_class = ChatListSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        return Chat.objects.filter(user=self.request.user, status=Chat.ChatStatus.ACTIVE).order_by('-last_message_at')


class ArchivedChatListView(generics.ListAPIView):
    # ... (sem alterações) ...
    serializer_class = ChatListSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        bot_id = self.kwargs['bot_id']
        return Chat.objects.filter(
            user=self.request.user,
            bot_id=bot_id,
            status=Chat.ChatStatus.ARCHIVED
        ).order_by('-last_message_at')

class ChatBootstrapView(APIView):
    # ... (sem alterações) ...
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request, bot_id):
        bot = get_object_or_404(Bot, id=bot_id)
        active_chat = Chat.objects.filter(
            user=request.user,
            bot=bot,
            status=Chat.ChatStatus.ACTIVE
        ).first()
        if not active_chat:
            active_chat = Chat.objects.create(
                user=request.user,
                bot=bot,
                status=Chat.ChatStatus.ACTIVE
            )
        chat = active_chat
        suggestions = [s for s in [bot.suggestion1, bot.suggestion2, bot.suggestion3] if s]
        avatar_url_path = None
        if bot.avatar_url and hasattr(bot.avatar_url, 'url'):
             try:
                  avatar_url_path = request.build_absolute_uri(bot.avatar_url.url)
             except Exception:
                  avatar_url_path = bot.avatar_url.url # Fallback para URL relativa

        bootstrap_data = {
            "conversationId": str(chat.id),
            "bot": { "name": bot.name, "handle": f"@{bot.owner.username}", "avatarUrl": avatar_url_path },
            "welcome": bot.description or "Hello! How can I help you today?",
            "suggestions": suggestions
        }
        return Response(bootstrap_data, status=status.HTTP_200_OK)


class ChatMessageListView(generics.ListCreateAPIView):
    # Serializer padrão para GET (list) e POST (create de texto)
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardMessagePagination

    def get_queryset(self):
        chat_id = self.kwargs['chat_pk']
        # Verifica se o chat existe e pertence ao usuário
        get_object_or_404(Chat, id=chat_id, user=self.request.user)
        return ChatMessage.objects.filter(chat_id=chat_id).order_by('-created_at')

    # Este 'create' agora é SÓ para mensagens de TEXTO (JSON)
    def create(self, request, *args, **kwargs):
        # Garante que esta view SÓ aceite application/json
        if not request.content_type or 'application/json' not in request.content_type.lower():
             return Response({"detail": "Content-Type must be application/json for text messages."},
                            status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

        # Usa o serializer padrão (ChatMessageSerializer)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)

        if chat.status != Chat.ChatStatus.ACTIVE:
            return Response({"detail": "This chat is archived and read-only."}, status=status.HTTP_403_FORBIDDEN)

        # Salva a mensagem do usuário (SEM anexo aqui)
        user_message = serializer.save(chat=chat, role=ChatMessage.Role.USER)
        chat.last_message_at = timezone.now() # Atualiza timestamp
        chat.save()

        # Obtém resposta da IA (igual a antes)
        ai_response_data = get_ai_response(chat_id, user_message.content)
        ai_content = ai_response_data.get('content')
        ai_suggestions = ai_response_data.get('suggestions', [])

        paragraphs = re.split(r'\n{2,}', ai_content.strip()) if ai_content else []

        ai_messages = []
        for i, paragraph_content in enumerate(paragraphs):
            is_first_paragraph = i == 0
            suggestions = ai_suggestions if is_first_paragraph else []

            ai_message = ChatMessage.objects.create(
                chat=chat,
                role=ChatMessage.Role.ASSISTANT,
                content=paragraph_content,
                suggestion1=suggestions[0] if len(suggestions) > 0 else None,
                suggestion2=suggestions[1] if len(suggestions) > 1 else None,
            )
            ai_messages.append(ai_message)

        if ai_messages:
            chat.last_message_at = ai_messages[-1].created_at
            chat.save()

        all_new_messages = [user_message] + ai_messages
        # Serializa a lista completa usando o serializer padrão (que inclui attachment_url=null)
        response_serializer = self.get_serializer(all_new_messages, many=True, context={'request': request})
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


# --- NOVA VIEW PARA UPLOAD DE ANEXOS ---
class ChatMessageAttachmentView(generics.CreateAPIView):
    serializer_class = ChatMessageAttachmentSerializer # Usa o serializer de escrita
    permission_classes = [permissions.IsAuthenticated]
    # Especifica que aceita multipart/form-data
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def perform_create(self, serializer):
        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)

        if chat.status != Chat.ChatStatus.ACTIVE:
            # Usa DRFValidationError para retornar 400 Bad Request
            raise DRFValidationError({"detail": "This chat is archived and cannot receive attachments."})

        uploaded_file = self.request.FILES.get('attachment')
        if not uploaded_file:
            raise DRFValidationError({"attachment": "No file was submitted."})

        # Determinar o tipo
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        attachment_type = 'image' if mime_type and mime_type.startswith('image/') else 'file'

        # Validar tamanho (exemplo: 10MB) - Boa prática ter na view também
        MAX_UPLOAD_SIZE = 10 * 1024 * 1024
        if uploaded_file.size > MAX_UPLOAD_SIZE:
             raise DRFValidationError({"attachment": f"File size cannot exceed {MAX_UPLOAD_SIZE // (1024*1024)}MB."})

        # Salva a mensagem via serializer
        try:
            message = serializer.save(
                chat=chat,
                role=ChatMessage.Role.USER,
                # attachment já está sendo tratado pelo serializer
                attachment_type=attachment_type,
                original_filename=uploaded_file.name
            )
            # Atualiza o timestamp do chat APÓS salvar a mensagem
            chat.last_message_at = message.created_at
            chat.save()
            return message # Retorna a instância salva para create()
        except DjangoValidationError as e: # Captura erros de validação do modelo/FileField
            raise DRFValidationError(e.message_dict)
        except Exception as e: # Outros erros inesperados
             print(f"Error saving attachment message: {e}")
             raise DRFValidationError({"detail": "Failed to save attachment message."}, code=status.HTTP_500_INTERNAL_SERVER_ERROR)


    def create(self, request, *args, **kwargs):
        try:
            # O serializer valida os dados (presença do 'attachment')
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            # perform_create é chamado aqui e salva a instância
            instance = self.perform_create(serializer)
            # Re-serializa a instância criada com o serializer de LEITURA para incluir a URL
            read_serializer = ChatMessageSerializer(instance, context={'request': request})
            headers = self.get_success_headers(read_serializer.data)
            return Response(read_serializer.data, status=status.HTTP_201_CREATED, headers=headers)
        except DRFValidationError as e:
            # Retorna erros de validação (tamanho, chat arquivado, etc.)
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)
        # except Exception as e: # Captura outros erros inesperados (já tratados em perform_create)
        #     print(f"Unexpected error in ChatMessageAttachmentView.create: {e}")
        #     return Response({"detail": "An unexpected error occurred during file upload."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ArchiveChatView(APIView):
    # ... (sem alterações) ...
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request, chat_id):
        chat_to_archive = get_object_or_404(Chat, id=chat_id, user=request.user)
        if chat_to_archive.status != Chat.ChatStatus.ACTIVE:
             return Response({"detail": "Only active chats can be archived."}, status=status.HTTP_400_BAD_REQUEST)
        chat_to_archive.status = Chat.ChatStatus.ARCHIVED
        chat_to_archive.save()
        new_active_chat = Chat.objects.create(
            user=request.user,
            bot=chat_to_archive.bot,
            status=Chat.ChatStatus.ACTIVE
        )
        return Response({"new_chat_id": new_active_chat.id}, status=status.HTTP_201_CREATED)

class SetActiveChatView(APIView):
    # ... (sem alterações) ...
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request, chat_id):
        user = request.user
        chat_to_activate = get_object_or_404(Chat, id=chat_id, user=user)
        bot = chat_to_activate.bot
        if chat_to_activate.status == Chat.ChatStatus.ACTIVE:
            return Response({"status": "already active"}, status=status.HTTP_200_OK)
        current_active_chat = Chat.objects.filter(
            user=user,
            bot=bot,
            status=Chat.ChatStatus.ACTIVE
        ).first()
        if current_active_chat:
            current_active_chat.status = Chat.ChatStatus.ARCHIVED
            current_active_chat.save()
        chat_to_activate.status = Chat.ChatStatus.ACTIVE
        chat_to_activate.last_message_at = timezone.now()
        chat_to_activate.save()
        serializer = ChatListSerializer(chat_to_activate, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)