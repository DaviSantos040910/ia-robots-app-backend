# chat/views.py
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Chat, ChatMessage
from .serializers import ChatSerializer, ChatMessageSerializer
from bots.models import Bot
from django.shortcuts import get_object_or_404
from .ai_service import get_ai_response

class ChatListCreateView(generics.ListCreateAPIView):
    """
    API view for listing and creating chat sessions.
    """
    queryset = Chat.objects.all()
    serializer_class = ChatSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Users can only see their own chats.
        return Chat.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # When creating a chat, you need to specify the bot.
        bot_id = self.request.data.get('bot_id')
        bot = get_object_or_404(Bot, id=bot_id)
        serializer.save(user=self.request.user, bot=bot)

class ChatBootstrapView(APIView):
    """
    Provides the initial data needed to bootstrap the chat screen.
    It finds or creates a chat session and returns bot details and welcome message.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, bot_id):
        bot = get_object_or_404(Bot, id=bot_id)
        
        # Find an existing chat or create a new one
        chat, created = Chat.objects.get_or_create(user=request.user, bot=bot)

        # Structure the data exactly as the frontend's ChatBootstrap type expects
        bootstrap_data = {
            "conversationId": str(chat.id),
            "bot": {
                "name": bot.name,
                "handle": f"@{bot.owner.username}",
                "avatarUrl": bot.avatar_url
            },
            "welcome": "Hello. I'm your new friend. You can ask me any questions.", # Placeholder message
            "suggestions": [ # Placeholder suggestions
                'Customize a savings plan for me.',
                'Have a healthy meal.',
                'U.S. travel plans for 2024.',
            ]
        }
        return Response(bootstrap_data, status=status.HTTP_200_OK)

class ChatMessageListCreateView(generics.ListCreateAPIView):
    """
    API view for listing and creating chat messages within a specific chat.
    Integrates with an AI service for bot responses.
    """
    queryset = ChatMessage.objects.all()
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        chat_id = self.kwargs['chat_pk']
        return ChatMessage.objects.filter(chat_id=chat_id, chat__user=self.request.user)

    def create(self, request, *args, **kwargs):
        chat_id = self.kwargs['chat_pk']
        chat = get_object_or_404(Chat, id=chat_id, user=self.request.user)
        user_message_content = request.data.get('content')

        if not user_message_content:
            return Response({"detail": "Content is required."}, status=status.HTTP_400_BAD_REQUEST)

        # 1. Save the user's message
        ChatMessage.objects.create(
            chat=chat, 
            role=ChatMessage.Role.USER, 
            content=user_message_content
        )

        # 2. Get the response from the AI service
        ai_response_content = get_ai_response(chat_id, user_message_content)

        # 3. Save the AI's message
        ai_message = ChatMessage.objects.create(
            chat=chat, 
            role=ChatMessage.Role.ASSISTANT, 
            content=ai_response_content
        )
        ai_message_serializer = self.get_serializer(ai_message)

        # 4. Return the AI's message to the frontend
        return Response(ai_message_serializer.data, status=status.HTTP_201_CREATED)