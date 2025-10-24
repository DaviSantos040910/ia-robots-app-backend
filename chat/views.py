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
import re # Importar a biblioteca de expressões regulares

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
        )

class ChatBootstrapView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, bot_id):
        bot = get_object_or_404(Bot, id=bot_id)
        chat, created = Chat.objects.get_or_create(
            user=request.user, 
            bot=bot, 
            status=Chat.ChatStatus.ACTIVE
        )
        suggestions = [s for s in [bot.suggestion1, bot.suggestion2, bot.suggestion3] if s]
        
        avatar_url_path = None
        if bot.avatar_url:
            # Construir o URL absoluto para o ficheiro de média
            avatar_url_path = request.build_absolute_uri(bot.avatar_url.url)
        
        bootstrap_data = {
            "conversationId": str(chat.id),
            "bot": { "name": bot.name, "handle": f"@{bot.owner.username}", "avatarUrl": avatar_url_path },
            "welcome": bot.description or "Hello! How can I help you today?",
            "suggestions": suggestions
        }
        return Response(bootstrap_data, status=status.HTTP_200_OK)

class ChatMessageListView(generics.ListCreateAPIView):
    """
    Lista mensagens com paginação (GET) e cria novas mensagens (POST).
    """
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardMessagePagination

    def get_queryset(self):
        chat_id = self.kwargs['chat_pk']
        get_object_or_404(Chat, id=chat_id, user=self.request.user)
        return ChatMessage.objects.filter(chat_id=chat_id).order_by('-created_at')

    # --- CORREÇÃO: 'create' movido para fora de 'get_queryset' e 'perform_create' removido ---
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True) # Valida o 'content'
        
        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)
        
        # Guardar a mensagem do utilizador
        user_message = serializer.save(chat=chat, role=ChatMessage.Role.USER)
        chat.last_message_at = timezone.now()
        chat.save()

        # Obter a resposta estruturada da IA (que é um dicionário)
        ai_response_data = get_ai_response(chat_id, user_message.content)
        ai_content = ai_response_data.get('content')
        ai_suggestions = ai_response_data.get('suggestions', [])
        
        # --- LÓGICA DE DIVISÃO DE MENSAGEM ---
        # Dividir a resposta completa em parágrafos (por duas ou mais quebras de linha)
        paragraphs = re.split(r'\n{2,}', ai_content.strip())
        
        ai_messages = [] # Uma lista para guardar as novas mensagens
        for i, paragraph_content in enumerate(paragraphs):
            # Apenas a primeira mensagem do bloco recebe as sugestões
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

        # Atualizar o timestamp do chat com a hora da última mensagem criada
        if ai_messages:
            chat.last_message_at = ai_messages[-1].created_at
            chat.save()

        # Serializar a LISTA de novas mensagens e retorná-la
        ai_serializer = self.get_serializer(ai_messages, many=True)
        return Response(ai_serializer.data, status=status.HTTP_201_CREATED)

class ArchiveChatView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, chat_id):
        chat_to_archive = get_object_or_404(Chat, id=chat_id, user=request.user)
        chat_to_archive.status = Chat.ChatStatus.ARCHIVED
        chat_to_archive.save()
        new_active_chat = Chat.objects.create(
            user=request.user,
            bot=chat_to_archive.bot,
            status=Chat.ChatStatus.ACTIVE
        )
        return Response({"new_chat_id": new_active_chat.id}, status=status.HTTP_201_CREATED)
class SetActiveChatView(APIView):
    """
    API view to set a specific (usually archived) chat as the active one for a bot.
    This implicitly archives the currently active chat for the same bot.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, chat_id):
        """
        Handles the POST request to activate a chat.
        """
        user = request.user
        
        # 1. Find the chat the user wants to activate
        #    Ensure it belongs to the user
        chat_to_activate = get_object_or_404(Chat, id=chat_id, user=user)
        bot = chat_to_activate.bot

        # If it's already active, there's nothing to do
        if chat_to_activate.status == Chat.ChatStatus.ACTIVE:
            return Response({"status": "already active"}, status=status.HTTP_200_OK)

        # 2. Find the currently active chat for the *same bot* and *same user* (if it exists)
        current_active_chat = Chat.objects.filter(
            user=user, 
            bot=bot, 
            status=Chat.ChatStatus.ACTIVE
        ).first() # Use .first() as there should ideally be only one

        # 3. Archive the (old) currently active chat
        if current_active_chat:
            current_active_chat.status = Chat.ChatStatus.ARCHIVED
            current_active_chat.save()

        # 4. Activate the selected chat
        chat_to_activate.status = Chat.ChatStatus.ACTIVE
        # Update the timestamp so it appears at the top of the main chat list
        chat_to_activate.last_message_at = timezone.now() 
        chat_to_activate.save()

        # Return the newly activated chat's data
        serializer = ChatListSerializer(chat_to_activate) 
        return Response(serializer.data, status=status.HTTP_200_OK)