# chat/views.py

from rest_framework import generics, permissions, status, parsers
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from .models import Chat, ChatMessage
from django.conf import settings
import os

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
    """
    Retorna dados iniciais do chat:
    - Informações do bot
    - ID da conversa ativa (ou cria uma nova)
    - Mensagem de boas-vindas
    - Sugestões iniciais
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, bot_id):
        bot = get_object_or_404(Bot, id=bot_id)

        # Busca ou cria um chat ativo para este bot
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

        # Constrói URL do avatar
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
    """
    GET: Lista mensagens do chat com paginação
    POST: Cria uma nova mensagem de texto e obtém resposta da IA
    """
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardMessagePagination

    def get_queryset(self):
        chat_id = self.kwargs['chat_pk']
        get_object_or_404(Chat, id=chat_id, user=self.request.user)
        return ChatMessage.objects.filter(chat_id=chat_id).order_by('-created_at')

    def create(self, request, *args, **kwargs):
        """Cria mensagem de texto e obtém resposta da IA"""
        # Valida Content-Type
        if not request.content_type or 'application/json' not in request.content_type.lower():
            return Response(
                {"detail": "Content-Type must be application/json for text messages."},
                status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            )

        # Valida dados
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)

        # Verifica se o chat está ativo
        if chat.status != Chat.ChatStatus.ACTIVE:
            return Response(
                {"detail": "This chat is archived and read-only."},
                status=status.HTTP_403_FORBIDDEN
            )

        # Salva a mensagem do usuário
        user_message = serializer.save(chat=chat, role=ChatMessage.Role.USER)
        chat.last_message_at = timezone.now()
        chat.save()

        # Obtém resposta da IA (passando o objeto da mensagem de texto)
        ai_response_data = get_ai_response(chat_id, user_message.content, user_message_obj=user_message)
        ai_content = ai_response_data.get('content')
        ai_suggestions = ai_response_data.get('suggestions', [])

        # Divide a resposta em parágrafos
        paragraphs = re.split(r'\n{2,}', ai_content.strip()) if ai_content else []
        if not paragraphs:
            paragraphs = ["..."]

        # Cria mensagens de resposta da IA
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

        # Atualiza timestamp do chat
        if ai_messages:
            chat.last_message_at = ai_messages[-1].created_at
            chat.save()

        # Retorna todas as mensagens criadas
        all_new_messages = [user_message] + ai_messages
        response_serializer = self.get_serializer(
            all_new_messages, 
            many=True, 
            context={'request': request}
        )

        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


# chat/views.py

class ChatMessageAttachmentView(generics.CreateAPIView):
    """
    POST: Upload de anexos (imagens/arquivos) com legenda opcional
    Processa o anexo e obtém resposta da IA
    """
    serializer_class = ChatMessageAttachmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def perform_create(self, serializer):
        """Valida e salva a mensagem com anexo"""
        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)

        if chat.status != Chat.ChatStatus.ACTIVE:
            raise DRFValidationError({
                "detail": "This chat is archived and cannot receive attachments."
            })

        uploaded_file = self.request.FILES.get('attachment')
        if not uploaded_file:
            raise DRFValidationError({"attachment": "No file was submitted."})

        # Determina o tipo do arquivo
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        attachment_type = 'image' if mime_type and mime_type.startswith('image/') else 'file'

        # Valida tamanho (10MB máximo)
        MAX_UPLOAD_SIZE = 10 * 1024 * 1024
        if uploaded_file.size > MAX_UPLOAD_SIZE:
            raise DRFValidationError({
                "attachment": f"File size cannot exceed {MAX_UPLOAD_SIZE // (1024*1024)}MB."
            })

        try:
            # Salva a mensagem
            message = serializer.save(
                chat=chat,
                role=ChatMessage.Role.USER,
                attachment_type=attachment_type,
                original_filename=uploaded_file.name
            )

            # Atualiza timestamp do chat
            chat.last_message_at = message.created_at
            chat.save()

            return message

        except DjangoValidationError as e:
            raise DRFValidationError(e.message_dict)
        except Exception as e:
            print(f"Error saving attachment message: {e}")
            raise DRFValidationError(
                {"detail": "Failed to save attachment message."},
                code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def create(self, request, *args, **kwargs):
        """
        Processa o upload, mas NÃO obtém resposta da IA.
        A IA só será chamada quando uma mensagem de TEXTO for enviada.
        """
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            user_message = self.perform_create(serializer)

            # ✅ CORREÇÃO CRÍTICA: Passar o 'context' para o serializador
            # Isso garante que o 'get_attachment_url' possa construir a URL completa
            read_serializer = ChatMessageSerializer(
                user_message,
                context={'request': request}  # <--- ESSENCIAL PARA URLs COMPLETAS
            )

            headers = self.get_success_headers(read_serializer.data)
            print(f"[ChatMessageAttachmentView] Anexo salvo. ID: {user_message.id}. URL: {read_serializer.data.get('attachment_url')}")

            # Retorna a mensagem do usuário como um array
            return Response(
                [read_serializer.data],
                status=status.HTTP_201_CREATED,
                headers=headers
            )

        except DRFValidationError as e:
            print(f"[ChatMessageAttachmentView] Validation error: {e.detail}")
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"[ChatMessageAttachmentView] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                {"detail": "An error occurred while processing the attachment."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AudioTranscriptionView(APIView):
    """
    POST: Transcreve um arquivo de áudio usando Google Gemini API
    Retorna o texto transcrito para ser editado antes de enviar
    """
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def post(self, request, chat_pk):
        """Recebe áudio e retorna transcrição"""
        try:
            # Valida que o chat existe e pertence ao usuário
            chat = get_object_or_404(Chat, id=chat_pk, user=request.user)

            if chat.status != Chat.ChatStatus.ACTIVE:
                return Response(
                    {"detail": "This chat is archived and read-only."},
                    status=status.HTTP_403_FORBIDDEN
                )

            # Obtém o arquivo de áudio
            audio_file = request.FILES.get('audio')
            if not audio_file:
                return Response(
                    {"detail": "No audio file provided."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            print(f"[AudioTranscriptionView] Received audio file: {audio_file.name}")
            print(f"[AudioTranscriptionView] File size: {audio_file.size} bytes")

            # Valida tamanho do arquivo (20MB máximo para Gemini)
            MAX_AUDIO_SIZE = 20 * 1024 * 1024
            if audio_file.size > MAX_AUDIO_SIZE:
                return Response(
                    {"detail": f"Audio file too large. Maximum size is {MAX_AUDIO_SIZE // (1024*1024)}MB."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Chama serviço de transcrição do Gemini
            print("[AudioTranscriptionView] Starting transcription with Gemini...")
            transcription_result = transcribe_audio_gemini(audio_file)

            if not transcription_result['success']:
                error_msg = transcription_result.get('error', 'Unknown error')
                print(f"[AudioTranscriptionView] Transcription failed: {error_msg}")
                return Response(
                    {"detail": f"Failed to transcribe audio: {error_msg}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            transcription_text = transcription_result['transcription']
            print(f"[AudioTranscriptionView] Transcription successful: {transcription_text[:100]}...")

            return Response(
                {"transcription": transcription_text},
                status=status.HTTP_200_OK
            )

        except Exception as e:
            print(f"[AudioTranscriptionView] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                {"detail": "An error occurred while transcribing audio."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ArchiveChatView(APIView):
    """Arquiva o chat atual e cria um novo chat ativo"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, chat_id):
        chat_to_archive = get_object_or_404(Chat, id=chat_id, user=request.user)

        if chat_to_archive.status != Chat.ChatStatus.ACTIVE:
            return Response(
                {"detail": "Only active chats can be archived."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Arquiva o chat atual
        chat_to_archive.status = Chat.ChatStatus.ARCHIVED
        chat_to_archive.save()

        # Cria novo chat ativo
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
    """Define um chat arquivado como ativo (arquivando o atual)"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, chat_id):
        user = request.user
        chat_to_activate = get_object_or_404(Chat, id=chat_id, user=user)
        bot = chat_to_activate.bot

        # Se já está ativo, retorna sucesso
        if chat_to_activate.status == Chat.ChatStatus.ACTIVE:
            return Response({"status": "already active"}, status=status.HTTP_200_OK)

        # Arquiva o chat atualmente ativo (se existir)
        current_active_chat = Chat.objects.filter(
            user=user,
            bot=bot,
            status=Chat.ChatStatus.ACTIVE
        ).first()

        if current_active_chat:
            current_active_chat.status = Chat.ChatStatus.ARCHIVED
            current_active_chat.save()

        # Ativa o chat selecionado
        chat_to_activate.status = Chat.ChatStatus.ACTIVE
        chat_to_activate.last_message_at = timezone.now()
        chat_to_activate.save()

        serializer = ChatListSerializer(chat_to_activate, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class CleanupFileResponse(FileResponse):
    """
    FileResponse customizado que deleta o arquivo temporário após o envio
    """
    def __init__(self, *args, cleanup_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_path = cleanup_path

    def close(self):
        """Remove o arquivo após o envio"""
        super().close()
        if self.cleanup_path and os.path.exists(self.cleanup_path):
            try:
                os.remove(self.cleanup_path)
                print(f"[TTS Cleanup] Arquivo removido com sucesso: {self.cleanup_path}")
            except Exception as e:
                print(f"[TTS Cleanup] Erro ao remover arquivo: {e}")


class MessageTTSView(APIView):
    """
    Gera e retorna áudio TTS para uma mensagem específica
    O arquivo é temporário e será deletado após o envio
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, chat_pk, message_id):
        # Valida que a mensagem existe e pertence ao usuário
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

        # Cria diretório temporário
        temp_dir = Path(settings.MEDIA_ROOT) / 'temp_tts'
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Gera nome único para o arquivo
        audio_filename = f"tts_{message.id}_{uuid.uuid4().hex[:8]}.wav"
        audio_path = temp_dir / audio_filename

        try:
            print(f"[MessageTTSView] Gerando TTS para mensagem {message_id}")

            # Gera o áudio
            result = generate_tts_audio(message.content, str(audio_path))

            if not result['success']:
                return Response(
                    {"detail": f"Erro ao gerar áudio: {result.get('error', 'Unknown error')}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Abre e envia o arquivo
            audio_file = open(audio_path, 'rb')
            response = CleanupFileResponse(
                audio_file,
                content_type='audio/wav',
                as_attachment=False,
                filename=audio_filename,
                cleanup_path=str(audio_path)
            )

            print(f"[MessageTTSView] Enviando arquivo TTS: {audio_path}")
            return response

        except Exception as e:
            # Limpa arquivo em caso de erro
            if audio_path.exists():
                try:
                    audio_path.unlink()
                    print(f"[MessageTTSView] Arquivo removido após erro: {audio_path}")
                except Exception as cleanup_error:
                    print(f"[MessageTTSView] Erro ao remover arquivo: {cleanup_error}")

            print(f"[MessageTTSView] Erro: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                {"detail": "Erro ao processar solicitação de áudio."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class MessageLikeToggleView(APIView):
    """Alterna o status de curtida de uma mensagem"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, chat_pk, message_id):
        message = get_object_or_404(
            ChatMessage,
            id=message_id,
            chat_id=chat_pk,
            chat__user=request.user
        )

        # Alterna o estado de curtida
        message.liked = not message.liked
        message.save()

        return Response({'liked': message.liked}, status=status.HTTP_200_OK)