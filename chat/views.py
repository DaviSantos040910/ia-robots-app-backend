# chat/views.py

from rest_framework import generics, permissions, status, parsers
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from .models import Chat, ChatMessage
from django.conf import settings
import os
from django.db import transaction

from .serializers import (
    ChatListSerializer, ChatMessageSerializer, ChatMessageAttachmentSerializer
)

from bots.models import Bot
from django.shortcuts import get_object_or_404
from .ai_service import get_ai_response, transcribe_audio_gemini, generate_tts_audio
from myproject.pagination import StandardMessagePagination
import re
import mimetypes
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework.exceptions import ValidationError as DRFValidationError
from django.http import FileResponse
from pathlib import Path
import uuid


class ActiveChatListView(generics.ListAPIView):
    """Lista todos os chats ativos do usuário"""
    serializer_class = ChatListSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Chat.objects.filter(
            user=self.request.user, 
            status=Chat.ChatStatus.ACTIVE
        ).order_by('-last_message_at')


class ArchivedChatListView(generics.ListAPIView):
    """Lista todos os chats arquivados de um bot específico"""
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
                avatar_url_path = bot.avatar_url.url

        bootstrap_data = {
            "conversationId": str(chat.id),
            "bot": {
                "name": bot.name,
                "handle": f"@{bot.owner.username}",
                "avatarUrl": avatar_url_path
            },
            "welcome": bot.description or "Hello! How can I help you today?",
            "suggestions": suggestions
        }

        return Response(bootstrap_data, status=status.HTTP_200_OK)


class ChatMessageListView(generics.ListCreateAPIView):
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardMessagePagination

    def get_queryset(self):
        chat_id = self.kwargs['chat_pk']
        get_object_or_404(Chat, id=chat_id, user=self.request.user)
        return ChatMessage.objects.filter(chat_id=chat_id).order_by('-created_at')

    def create(self, request, *args, **kwargs):
        if not request.content_type or 'application/json' not in request.content_type.lower():
            return Response(
                {"detail": "Content-Type must be application/json for text messages."},
                status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            )

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)

        if chat.status != Chat.ChatStatus.ACTIVE:
            return Response(
                {"detail": "This chat is archived and read-only."},
                status=status.HTTP_403_FORBIDDEN
            )

        user_message = serializer.save(chat=chat, role=ChatMessage.Role.USER)
        chat.last_message_at = timezone.now()
        chat.save()

        ai_response_data = get_ai_response(chat_id, user_message.content, user_message_obj=user_message)
        ai_content = ai_response_data.get('content')
        ai_suggestions = ai_response_data.get('suggestions', [])

        paragraphs = re.split(r'\n{2,}', ai_content.strip()) if ai_content else []
        if not paragraphs:
            paragraphs = ["..."]

        ai_messages = []
        total_paragraphs = len(paragraphs)

        for i, paragraph_content in enumerate(paragraphs):
            is_last_paragraph = i == (total_paragraphs - 1)
            suggestions = ai_suggestions if is_last_paragraph else []

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
        response_serializer = self.get_serializer(
            all_new_messages, 
            many=True, 
            context={'request': request}
        )

        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


class ChatMessageAttachmentView(generics.CreateAPIView):
    """
    POST: Upload de MÚLTIPLOS anexos (batch upload).
    Aceita chave 'attachments' contendo múltiplos arquivos.
    """
    serializer_class = ChatMessageAttachmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def create(self, request, *args, **kwargs):
        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)

        if chat.status != Chat.ChatStatus.ACTIVE:
            return Response(
                {"detail": "This chat is archived and cannot receive attachments."},
                status=status.HTTP_403_FORBIDDEN
            )

        # 1. Obter a lista de arquivos.
        # O frontend deve enviar FormData com múltiplos campos chamados 'attachments'
        files = request.FILES.getlist('attachments')
        
        # Fallback para 'attachment' singular (retrocompatibilidade)
        if not files and request.FILES.get('attachment'):
            files = [request.FILES.get('attachment')]

        if not files:
            return Response(
                {"detail": "No files provided. Use key 'attachments'."}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        created_messages = []
        
        # 2. Processamento em Batch com Atomicidade
        try:
            with transaction.atomic():
                for uploaded_file in files:
                    # Validação de tamanho individual (ex: 10MB)
                    MAX_UPLOAD_SIZE = 10 * 1024 * 1024
                    if uploaded_file.size > MAX_UPLOAD_SIZE:
                         raise DRFValidationError(f"File {uploaded_file.name} exceeds 10MB limit.")

                    # Determina tipo
                    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
                    attachment_type = 'image' if mime_type and mime_type.startswith('image/') else 'file'

                    # Prepara dados para o serializer
                    data = {
                        'attachment': uploaded_file,
                        # Se tiver uma legenda única para todos, pode vir em request.data.get('content')
                        # Aqui deixamos vazio para focar no anexo
                        'content': '' 
                    }
                    
                    serializer = self.get_serializer(data=data)
                    serializer.is_valid(raise_exception=True)
                    
                    # Salva a mensagem
                    message = serializer.save(
                        chat=chat,
                        role=ChatMessage.Role.USER,
                        attachment_type=attachment_type,
                        original_filename=uploaded_file.name
                    )
                    created_messages.append(message)

                # Atualiza timestamp do chat uma vez no final
                if created_messages:
                    chat.last_message_at = created_messages[-1].created_at
                    chat.save()

            # 3. Serializa resposta
            read_serializer = ChatMessageSerializer(
                created_messages,
                many=True,
                context={'request': request}
            )

            return Response(read_serializer.data, status=status.HTTP_201_CREATED)

        except DRFValidationError as e:
            return Response({"detail": e.detail}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"[Batch Upload Error] {e}")
            import traceback
            traceback.print_exc()
            return Response(
                {"detail": "Failed to process batch upload."}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AudioTranscriptionView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def post(self, request, chat_pk):
        try:
            chat = get_object_or_404(Chat, id=chat_pk, user=request.user)

            if chat.status != Chat.ChatStatus.ACTIVE:
                return Response(
                    {"detail": "This chat is archived and read-only."},
                    status=status.HTTP_403_FORBIDDEN
                )

            audio_file = request.FILES.get('audio')
            if not audio_file:
                return Response(
                    {"detail": "No audio file provided."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            MAX_AUDIO_SIZE = 20 * 1024 * 1024
            if audio_file.size > MAX_AUDIO_SIZE:
                return Response(
                    {"detail": f"Audio file too large. Maximum size is {MAX_AUDIO_SIZE // (1024*1024)}MB."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            transcription_result = transcribe_audio_gemini(audio_file)

            if not transcription_result['success']:
                error_msg = transcription_result.get('error', 'Unknown error')
                return Response(
                    {"detail": f"Failed to transcribe audio: {error_msg}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            transcription_text = transcription_result['transcription']
            return Response(
                {"transcription": transcription_text},
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {"detail": "An error occurred while transcribing audio."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ArchiveChatView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, chat_id):
        chat_to_archive = get_object_or_404(Chat, id=chat_id, user=request.user)

        if chat_to_archive.status != Chat.ChatStatus.ACTIVE:
            return Response(
                {"detail": "Only active chats can be archived."},
                status=status.HTTP_400_BAD_REQUEST
            )

        chat_to_archive.status = Chat.ChatStatus.ARCHIVED
        chat_to_archive.save()

        new_active_chat = Chat.objects.create(
            user=request.user,
            bot=chat_to_archive.bot,
            status=Chat.ChatStatus.ACTIVE
        )

        return Response(
            {"new_chat_id": new_active_chat.id},
            status=status.HTTP_201_CREATED
        )


class SetActiveChatView(APIView):
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


class CleanupFileResponse(FileResponse):
    def __init__(self, *args, cleanup_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_path = cleanup_path

    def close(self):
        super().close()
        if self.cleanup_path and os.path.exists(self.cleanup_path):
            try:
                os.remove(self.cleanup_path)
            except Exception as e:
                print(f"[TTS Cleanup] Erro ao remover arquivo: {e}")


class MessageTTSView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, chat_pk, message_id):
        message = get_object_or_404(
            ChatMessage,
            id=message_id,
            chat_id=chat_pk,
            chat__user=request.user,
            role=ChatMessage.Role.ASSISTANT
        )

        if not message.content:
            return Response(
                {"detail": "Esta mensagem não tem conteúdo de texto para gerar áudio."},
                status=status.HTTP_400_BAD_REQUEST
            )

        temp_dir = Path(settings.MEDIA_ROOT) / 'temp_tts'
        temp_dir.mkdir(parents=True, exist_ok=True)
        audio_filename = f"tts_{message.id}_{uuid.uuid4().hex[:8]}.wav"
        audio_path = temp_dir / audio_filename

        try:
            result = generate_tts_audio(message.content, str(audio_path))

            if not result['success']:
                return Response(
                    {"detail": f"Erro ao gerar áudio: {result.get('error', 'Unknown error')}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            audio_file = open(audio_path, 'rb')
            response = CleanupFileResponse(
                audio_file,
                content_type='audio/wav',
                as_attachment=False,
                filename=audio_filename,
                cleanup_path=str(audio_path)
            )
            return response

        except Exception as e:
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception:
                    pass
            return Response(
                {"detail": "Erro ao processar solicitação de áudio."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class MessageLikeToggleView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, chat_pk, message_id):
        message = get_object_or_404(
            ChatMessage,
            id=message_id,
            chat_id=chat_pk,
            chat__user=request.user
        )
        message.liked = not message.liked
        message.save()
        return Response({'liked': message.liked}, status=status.HTTP_200_OK)