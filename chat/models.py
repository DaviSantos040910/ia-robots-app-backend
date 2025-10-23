# chat/models.py
from django.db import models
from django.conf import settings
from bots.models import Bot

class Chat(models.Model):
    """
    Represents a chat session between a user and a bot.
    A user can have multiple chat sessions with the same bot, but only one can be 'active'.
    """
    class ChatStatus(models.TextChoices):
        ACTIVE = 'active', 'Active'
        ARCHIVED = 'archived', 'Archived'

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='chats')
    bot = models.ForeignKey(Bot, on_delete=models.CASCADE, related_name='chats')
    
    # --- NEW: Status field to manage active vs. archived conversations ---
    status = models.CharField(
        max_length=10,
        choices=ChatStatus.choices,
        default=ChatStatus.ACTIVE
    )
    
    # --- NEW: Timestamp for ordering the chat list ---
    last_message_at = models.DateTimeField(auto_now_add=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-last_message_at'] # Order chats by the most recent message

    def __str__(self):
        return f"Chat {self.id} between {self.user.username} and {self.bot.name} ({self.status})"

class ChatMessage(models.Model):
    """
    Represents a single message in a chat.
    """
    class Role(models.TextChoices):
        USER = 'user', 'User'
        ASSISTANT = 'assistant', 'Assistant'

    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=Role.choices)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    suggestion1 = models.CharField(max_length=128, null=True, blank=True, help_text="First follow-up suggestion.")
    suggestion2 = models.CharField(max_length=128, null=True, blank=True, help_text="Second follow-up suggestion.")
    # -----------------------
    def __str__(self):
        return f"{self.role}: {self.content[:50]}"