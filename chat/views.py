# chat/views.py
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from .models import Chat, ChatMessage
from .serializers import ChatListSerializer, ChatMessageSerializer
from bots.models import Bot
from django.shortcuts import get_object_or_404
from .ai_service import get_ai_response
from myproject.pagination import StandardMessagePagination
import re

class ActiveChatListView(generics.ListAPIView):
    serializer_class = ChatListSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        return Chat.objects.filter(user=self.request.user, status=Chat.ChatStatus.ACTIVE)

class ArchivedChatListView(generics.ListAPIView):
    serializer_class = ChatListSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        bot_id = self.kwargs['bot_id']
        return Chat.objects.filter(
            user=self.request.user, 
            bot_id=bot_id, 
            status=Chat.ChatStatus.ARCHIVED
        ).order_by('-last_message_at') # Ordena para mostrar as mais recentes primeiro

class ChatBootstrapView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, bot_id):
        bot = get_object_or_404(Bot, id=bot_id)
        
        # --- LÓGICA DE BOOTSTRAP ATUALIZADA ---
        # 1. Tenta encontrar um chat ATIVO
        active_chat = Chat.objects.filter(
            user=request.user, 
            bot=bot, 
            status=Chat.ChatStatus.ACTIVE
        ).first()
        
        # 2. Se não houver chat ativo, cria um novo
        if not active_chat:
            active_chat = Chat.objects.create(
                user=request.user, 
                bot=bot, 
                status=Chat.ChatStatus.ACTIVE
            )
            
        chat = active_chat # Usa o chat ativo encontrado ou criado
        # --- FIM DA LÓGICA ATUALIZADA ---

        suggestions = [s for s in [bot.suggestion1, bot.suggestion2, bot.suggestion3] if s]
        
        avatar_url_path = None
        if bot.avatar_url:
            avatar_url_path = request.build_absolute_uri(bot.avatar_url.url)
        
        bootstrap_data = {
            "conversationId": str(chat.id),
            "bot": { "name": bot.name, "handle": f"@{bot.owner.username}", "avatarUrl": avatar_url_path },
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
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)
        
        # --- CORREÇÃO: VERIFICA SE O CHAT ESTÁ ATIVO ANTES DE PERMITIR MENSAGEM ---
        if chat.status != Chat.ChatStatus.ACTIVE:
            return Response({"detail": "This chat is archived and read-only."}, status=status.HTTP_403_FORBIDDEN)
            
        user_message = serializer.save(chat=chat, role=ChatMessage.Role.USER)
        chat.last_message_at = timezone.now()
        chat.save()

        ai_response_data = get_ai_response(chat_id, user_message.content)
        ai_content = ai_response_data.get('content')
        ai_suggestions = ai_response_data.get('suggestions', [])
        
        paragraphs = re.split(r'\n{2,}', ai_content.strip())
        
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

        # --- CORREÇÃO PRINCIPAL: Retorna a user_message + ai_messages ---
        # Coloca a mensagem do usuário (agora com ID real) e as respostas da IA numa única lista
        all_new_messages = [user_message] + ai_messages
        
        # Serializa a LISTA completa
        ai_serializer = self.get_serializer(all_new_messages, many=True)
        return Response(ai_serializer.data, status=status.HTTP_201_CREATED)
        # --- FIM DA CORREÇÃO ---

class ArchiveChatView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, chat_id):
        chat_to_archive = get_object_or_404(Chat, id=chat_id, user=request.user)
        
        # Garante que só o chat ativo pode ser arquivado
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

        # --- CORREÇÃO: Retorna o objeto ChatListItem completo ---
        # O frontend espera os mesmos dados da lista de chat
        serializer = ChatListSerializer(chat_to_activate, context={'request': request}) 
        return Response(serializer.data, status=status.HTTP_200_OK)