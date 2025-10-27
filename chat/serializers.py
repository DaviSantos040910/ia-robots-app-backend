# chat/serializers.py
from rest_framework import serializers
from .models import Chat, ChatMessage
from bots.serializers import BotSerializer # We'll need this to represent the bot in the chat list

class ChatMessageSerializer(serializers.ModelSerializer):
    """
    Serializer for the ChatMessage model.
    """
    suggestions = serializers.SerializerMethodField()
    class Meta:
        model = ChatMessage
        fields = ('id', 'chat', 'role', 'content', 'created_at', 'suggestions')
        read_only_fields = ('id', 'chat', 'role', 'created_at', 'suggestions')
    def get_suggestions(self, obj):
        """
        Combines suggestion1 and suggestion2 into a list, filtering out empty ones.
        """
        suggestions_list = []
        if obj.suggestion1:
            suggestions_list.append(obj.suggestion1)
        if obj.suggestion2:
            suggestions_list.append(obj.suggestion2)
        return suggestions_list

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