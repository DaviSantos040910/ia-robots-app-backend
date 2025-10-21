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
from myproject.pagination import StandardMessagePagination # Import our pagination class

class ActiveChatListView(generics.ListAPIView):
    """
    --- MODIFIED VIEW ---
    Lists all ACTIVE chat sessions for the authenticated user.
    """
    serializer_class = ChatListSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Chat.objects.filter(user=self.request.user, status=Chat.ChatStatus.ACTIVE)

class ArchivedChatListView(generics.ListAPIView):
    """
    --- NEW VIEW ---
    Lists all ARCHIVED chat sessions for a specific bot and user.
    """
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
    """
    Finds the active chat for a bot or creates a new one.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, bot_id):
        bot = get_object_or_404(Bot, id=bot_id)
        
        # --- MODIFIED LOGIC ---
        # Find an existing ACTIVE chat or create a new one.
        chat, created = Chat.objects.get_or_create(
            user=request.user, 
            bot=bot, 
            status=Chat.ChatStatus.ACTIVE
        )
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
    """
    --- MODIFIED VIEW ---
    Lists and creates chat messages with pagination.
    """
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardMessagePagination # Apply pagination

    def get_queryset(self):
        chat_id = self.kwargs['chat_pk']
        # Ensure the user can only access their own chats
        get_object_or_404(Chat, id=chat_id, user=self.request.user)
        return ChatMessage.objects.filter(chat_id=chat_id).order_by('-created_at') # Order by newest first for pagination

        def create(self, request, *args, **kwargs):
            serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)
        
        user_message = serializer.save(chat=chat, role=ChatMessage.Role.USER)
        chat.last_message_at = timezone.now()
        chat.save()

        ai_response_content = get_ai_response(chat_id, user_message.content)
        ai_message = ChatMessage.objects.create(
            chat=chat, role=ChatMessage.Role.ASSISTANT, content=ai_response_content
        )
        chat.last_message_at = timezone.now()
        chat.save()

        # Retorna a resposta da IA, como o frontend espera
        ai_serializer = self.get_serializer(ai_message)
        return Response(ai_serializer.data, status=status.HTTP_201_CREATED)

    def perform_create(self, serializer):
        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)
        
        # Save user message
        user_message = serializer.save(chat=chat, role=ChatMessage.Role.USER)
        
        # Update chat's last message timestamp to bring it to the top of the list
        chat.last_message_at = user_message.created_at
        chat.save()

        # Get AI response
        ai_response_content = get_ai_response(chat_id, user_message.content)
        
        # Save AI message
        ai_message = ChatMessage.objects.create(
            chat=chat, 
            role=ChatMessage.Role.ASSISTANT, 
            content=ai_response_content
        )
        
        # Update timestamp again with the AI message for perfect ordering
        chat.last_message_at = ai_message.created_at
        chat.save()

class ArchiveChatView(APIView):
    """
    --- NEW VIEW ---
    Archives a chat and creates a new active one for the same bot.
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, chat_id):
        # Archive the current chat
        chat_to_archive = get_object_or_404(Chat, id=chat_id, user=request.user)
        chat_to_archive.status = Chat.ChatStatus.ARCHIVED
        chat_to_archive.save()

        # Create a new active chat for the same bot and user
        new_active_chat = Chat.objects.create(
            user=request.user,
            bot=chat_to_archive.bot,
            status=Chat.ChatStatus.ACTIVE
        )

        # Return the ID of the new chat so the frontend can navigate to it
        return Response({"new_chat_id": new_active_chat.id}, status=status.HTTP_201_CREATED)