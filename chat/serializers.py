# chat/serializers.py
from rest_framework import serializers
from .models import Chat, ChatMessage

class ChatMessageSerializer(serializers.ModelSerializer):
    """
    Serializer for the ChatMessage model.
    """
    # --- CORREÇÃO APLICADA AQUI ---
    # Força o ID a ser enviado como uma string para corresponder ao tipo do frontend.
    id = serializers.CharField(read_only=True)

    class Meta:
        model = ChatMessage
        fields = ('id', 'chat', 'role', 'content', 'created_at')

class ChatSerializer(serializers.ModelSerializer):
    """
    Serializer for the Chat model, including related messages.
    """
    messages = ChatMessageSerializer(many=True, read_only=True)

    class Meta:
        model = Chat
        fields = ('id', 'user', 'bot', 'created_at', 'messages')