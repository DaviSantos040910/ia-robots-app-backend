# chat/views.py

from rest_framework import generics, permissions, status, parsers
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from .models import Chat, ChatMessage
from django.conf import settings
import os
from django.db import transaction
from .serializers import ChatListSerializer, ChatMessageSerializer, ChatMessageAttachmentSerializer
from bots.models import Bot
from django.shortcuts import get_object_or_404
from .ai_service import get_ai_response, transcribe_audio_gemini, generate_tts_audio, handle_voice_interaction, handle_voice_message
from myproject.pagination import StandardMessagePagination
import re
import mimetypes
from rest_framework.exceptions import ValidationError as DRFValidationError
from django.http import FileResponse
from pathlib import Path
import uuid
from django.core.files import File

# ... (ActiveChatListView, ArchivedChatListView, ChatBootstrapView permanecem iguais) ...
class ActiveChatListView(generics.ListAPIView):
    serializer_class = ChatListSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        return Chat.objects.filter(user=self.request.user, status=Chat.ChatStatus.ACTIVE).order_by('-last_message_at')

class ArchivedChatListView(generics.ListAPIView):
    serializer_class = ChatListSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        bot_id = self.kwargs['bot_id']
        return Chat.objects.filter(user=self.request.user, bot_id=bot_id, status=Chat.ChatStatus.ARCHIVED).order_by('-last_message_at')

class ChatBootstrapView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request, bot_id):
        bot = get_object_or_404(Bot, id=bot_id)
        active_chat = Chat.objects.filter(user=request.user, bot=bot, status=Chat.ChatStatus.ACTIVE).first()
        if not active_chat:
            active_chat = Chat.objects.create(user=request.user, bot=bot, status=Chat.ChatStatus.ACTIVE)
        
        avatar_url_path = None
        if bot.avatar_url and hasattr(bot.avatar_url, 'url'):
            try: avatar_url_path = request.build_absolute_uri(bot.avatar_url.url)
            except: avatar_url_path = bot.avatar_url.url

        return Response({
            "conversationId": str(active_chat.id),
            "bot": {"name": bot.name, "handle": f"@{bot.owner.username}", "avatarUrl": avatar_url_path},
            "welcome": bot.description or "Hello! How can I help you today?",
            "suggestions": [s for s in [bot.suggestion1, bot.suggestion2, bot.suggestion3] if s]
        }, status=status.HTTP_200_OK)

class ChatMessageListView(generics.ListCreateAPIView):
    """
    Lida com listagem e criação de mensagens de TEXTO.
    Agora suporta a flag 'reply_with_audio'.
    """
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardMessagePagination

    def get_queryset(self):
        chat_id = self.kwargs['chat_pk']
        get_object_or_404(Chat, id=chat_id, user=self.request.user)
        return ChatMessage.objects.filter(chat_id=chat_id).order_by('-created_at')

    def create(self, request, *args, **kwargs):
        if not request.content_type or 'application/json' not in request.content_type.lower():
            return Response({"detail": "Content-Type must be application/json for text messages."}, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)

        if chat.status != Chat.ChatStatus.ACTIVE:
            return Response({"detail": "This chat is archived and read-only."}, status=status.HTTP_403_FORBIDDEN)

        # --- NOVO: Captura a flag do corpo da requisição ---
        reply_with_audio = request.data.get('reply_with_audio', False)

        user_message = serializer.save(chat=chat, role=ChatMessage.Role.USER)
        chat.last_message_at = timezone.now()
        chat.save()

        # --- Passa a flag para o serviço de IA ---
        ai_response_data = get_ai_response(
            chat_id, 
            user_message.content, 
            user_message_obj=user_message,
            reply_with_audio=reply_with_audio # <--- Flag repassada
        )
        
        ai_content = ai_response_data.get('content')
        ai_suggestions = ai_response_data.get('suggestions', [])
        audio_path = ai_response_data.get('audio_path') # Caminho do arquivo gerado (se houver)

        # Tratamento de parágrafos (se houver)
        paragraphs = re.split(r'\n{2,}', ai_content.strip()) if ai_content else []
        if not paragraphs: paragraphs = ["..."]

        ai_messages = []
        total_paragraphs = len(paragraphs)

        for i, paragraph_content in enumerate(paragraphs):
            is_last_paragraph = i == (total_paragraphs - 1)
            suggestions = ai_suggestions if is_last_paragraph else []

            ai_message = ChatMessage(
                chat=chat,
                role=ChatMessage.Role.ASSISTANT,
                content=paragraph_content,
                suggestion1=suggestions[0] if len(suggestions) > 0 else None,
                suggestion2=suggestions[1] if len(suggestions) > 1 else None,
            )

            # --- NOVO: Se houver áudio gerado, anexa à última parte da resposta ---
            if is_last_paragraph and audio_path and os.path.exists(audio_path):
                try:
                    with open(audio_path, 'rb') as f:
                        # Gera um nome único para o arquivo
                        filename = f"reply_tts_{uuid.uuid4().hex[:10]}.wav"
                        ai_message.attachment.save(filename, File(f), save=False)
                        ai_message.attachment_type = 'audio'
                        ai_message.original_filename = "voice_reply.wav"
                    
                    # Limpa o arquivo temporário
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Erro ao anexar áudio TTS na view: {e}")

            ai_message.save()
            ai_messages.append(ai_message)

        if ai_messages:
            chat.last_message_at = ai_messages[-1].created_at
            chat.save()

        all_new_messages = [user_message] + ai_messages
        response_serializer = self.get_serializer(all_new_messages, many=True, context={'request': request})

        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


# ... (ChatMessageAttachmentView, AudioTranscriptionView, ArchiveChatView, SetActiveChatView, CleanupFileResponse, MessageTTSView, MessageLikeToggleView, VoiceInteractionView, VoiceMessageView permanecem iguais) ...
class ChatMessageAttachmentView(generics.CreateAPIView):
    serializer_class = ChatMessageAttachmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    def create(self, request, *args, **kwargs):
        # ... (implementação anterior mantida) ...
        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)
        if chat.status != Chat.ChatStatus.ACTIVE: return Response({"detail": "Archived."}, status=403)
        files = request.FILES.getlist('attachments') or ([request.FILES.get('attachment')] if request.FILES.get('attachment') else [])
        if not files: return Response({"detail": "No files."}, status=400)
        
        created_msgs = []
        try:
            with transaction.atomic():
                for f in files:
                    mime, _ = mimetypes.guess_type(f.name)
                    m = self.get_serializer(data={'attachment': f, 'content': ''})
                    m.is_valid(raise_exception=True)
                    obj = m.save(chat=chat, role=ChatMessage.Role.USER, attachment_type='image' if mime and mime.startswith('image/') else 'file', original_filename=f.name)
                    created_msgs.append(obj)
                if created_msgs:
                    chat.last_message_at = created_msgs[-1].created_at
                    chat.save()
            return Response(ChatMessageSerializer(created_msgs, many=True, context={'request':request}).data, status=201)
        except Exception as e: return Response({"detail": str(e)}, status=500)

class AudioTranscriptionView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    def post(self, request, chat_pk):
        chat = get_object_or_404(Chat, id=chat_pk, user=request.user)
        f = request.FILES.get('audio')
        if not f: return Response({"detail": "No audio."}, status=400)
        res = transcribe_audio_gemini(f)
        return Response({"transcription": res['transcription']} if res['success'] else {"detail": res['error']}, status=200 if res['success'] else 500)

class ArchiveChatView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request, chat_id):
        c = get_object_or_404(Chat, id=chat_id, user=request.user)
        c.status = Chat.ChatStatus.ARCHIVED; c.save()
        n = Chat.objects.create(user=request.user, bot=c.bot, status=Chat.ChatStatus.ACTIVE)
        return Response({"new_chat_id": n.id}, status=201)

class SetActiveChatView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request, chat_id):
        c = get_object_or_404(Chat, id=chat_id, user=request.user)
        Chat.objects.filter(user=request.user, bot=c.bot, status=Chat.ChatStatus.ACTIVE).update(status=Chat.ChatStatus.ARCHIVED)
        c.status = Chat.ChatStatus.ACTIVE; c.last_message_at = timezone.now(); c.save()
        return Response(ChatListSerializer(c, context={'request':request}).data, status=200)

class CleanupFileResponse(FileResponse):
    def __init__(self, *args, cleanup_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_path = cleanup_path
    def close(self):
        super().close()
        if self.cleanup_path and os.path.exists(self.cleanup_path):
            try: os.remove(self.cleanup_path)
            except: pass

class MessageTTSView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request, chat_pk, message_id):
        m = get_object_or_404(ChatMessage, id=message_id, chat_id=chat_pk, role=ChatMessage.Role.ASSISTANT)
        if not m.content: return Response({"detail": "No content"}, status=400)
        td = Path(settings.MEDIA_ROOT) / 'temp_tts'; td.mkdir(parents=True, exist_ok=True)
        p = td / f"tts_{m.id}_{uuid.uuid4().hex[:8]}.wav"
        res = generate_tts_audio(m.content, str(p))
        if res['success']: return CleanupFileResponse(open(p, 'rb'), content_type='audio/wav', cleanup_path=str(p))
        return Response({"detail": "Error"}, status=500)

class MessageLikeToggleView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request, chat_pk, message_id):
        m = get_object_or_404(ChatMessage, id=message_id, chat_id=chat_pk, chat__user=request.user)
        m.liked = not m.liked; m.save()
        return Response({'liked': m.liked}, status=200)

class VoiceInteractionView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    def post(self, request, chat_pk):
        chat = get_object_or_404(Chat, id=chat_pk, user=request.user)
        f = request.FILES.get('audio')
        if not f: return Response({"detail": "No audio."}, status=400)
        try:
            res = handle_voice_interaction(chat.id, f, request.user)
            return Response({
                "transcription": res['transcription'],
                "ai_response_text": res['ai_response_text'],
                "user_message": ChatMessageSerializer(res['user_message'], context={'request':request}).data,
                "ai_messages": ChatMessageSerializer(res['ai_messages'], many=True, context={'request':request}).data
            }, status=200)
        except Exception as e: return Response({"detail": str(e)}, status=500)

class VoiceMessageView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    def post(self, request, chat_pk):
        chat = get_object_or_404(Chat, id=chat_pk, user=request.user)
        f = request.FILES.get('audio') or request.FILES.get('file') or request.FILES.get('attachment')
        if not f: return Response({"detail": "No audio."}, status=400)
        reply_audio = str(request.data.get('reply_with_audio', 'false')).lower() == 'true'
        try:
            res = handle_voice_message(chat.id, f, reply_audio, request.user)
            return Response([
                ChatMessageSerializer(res['user_message'], context={'request':request}).data,
                ChatMessageSerializer(res['ai_message'], context={'request':request}).data
            ], status=201)
        except Exception as e: return Response({"detail": str(e)}, status=500)