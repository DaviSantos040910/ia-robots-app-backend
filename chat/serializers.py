# chat/serializers.py
from rest_framework import serializers
from .models import Chat, ChatMessage
from bots.serializers import BotSerializer # We'll need this to represent the bot in the chat list

class ChatMessageSerializer(serializers.ModelSerializer):
    """
    Serializer for the ChatMessage model.
    """
    class Meta:
        model = ChatMessage
        fields = ('id', 'chat', 'role', 'content', 'created_at')
        read_only_fields = ('id', 'chat', 'role', 'created_at')

class ChatListSerializer(serializers.ModelSerializer):
    """
    Serializer for the main Chat list screen.
    Includes details of the bot and a preview of the last message.
    """
    bot = BotSerializer(read_only=True)
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = Chat
        fields = ('id', 'bot', 'last_message', 'last_message_at', 'status')

    def get_last_message(self, obj):
        """
        Retrieves the most recent message for a given chat.
        """
        last_msg = obj.messages.order_by('-created_at').first()
        if last_msg:
            return ChatMessageSerializer(last_msg).data
        return None