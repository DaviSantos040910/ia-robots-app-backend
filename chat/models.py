# chat/models.py
from django.db import models
from django.conf import settings
from bots.models import Bot
import os # Importar os

def chat_attachment_path(instance, filename):
    # O ficheiro será guardado em MEDIA_ROOT/chat_attachments/chat_<id>/<filename>
    # Isso organiza os anexos por chat.
    # Garante que o filename é seguro (Django faz isso por padrão, mas reforça)
    filename = os.path.basename(filename) # Evita ataques de path traversal
    # Check if instance is ChatMessage or ChatFile
    if hasattr(instance, 'chat'):
        chat_id = instance.chat.id
    elif hasattr(instance, 'message') and instance.message.chat:
        chat_id = instance.message.chat.id
    else:
         # Fallback default if needed (though relations should be set)
        chat_id = 'unknown'

    return f'chat_attachments/chat_{chat_id}/{filename}'

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
    content = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Deprecated: usage of attachment directly in ChatMessage should be migrated to ChatFile
    attachment = models.FileField(
        upload_to=chat_attachment_path,
        null=True,
        blank=True,
        max_length=500
    )
    ATTACHMENT_TYPE_CHOICES = [
        ('image', 'Image'),
        ('file', 'File'),
        ('audio', 'Audio'),
    ]
    attachment_type = models.CharField(
        max_length=10,
        choices=ATTACHMENT_TYPE_CHOICES,
        null=True,
        blank=True
    )
    original_filename = models.CharField(max_length=255, null=True, blank=True)

    duration = models.IntegerField(default=0, help_text="Audio duration in milliseconds")

    # --------------------
    suggestion1 = models.CharField(max_length=128, null=True, blank=True, help_text="First follow-up suggestion.")
    suggestion2 = models.CharField(max_length=128, null=True, blank=True, help_text="Second follow-up suggestion.")
    liked = models.BooleanField(default=False)

    # -----------------------
    def __str__(self):
        if self.attachment and self.original_filename:
            return f"{self.role}: [Attachment: {self.original_filename}]"
        elif self.attachment:
             # Fallback se original_filename for nulo por algum motivo
             return f"{self.role}: [Attachment: {os.path.basename(self.attachment.name)}]"
        elif self.content:
             return f"{self.role}: {self.content[:50]}"
        else:
             return f"{self.role}: [Empty Message]" # Caso raro

    # Opcional: Limpar ficheiro ao apagar mensagem (Adapte para django-storages se usar nuvem)
    def delete(self, *args, **kwargs):
        storage = self.attachment.storage
        if storage and self.attachment.name and storage.exists(self.attachment.name):
            storage.delete(self.attachment.name)
        super().delete(*args, **kwargs)

class ChatFile(models.Model):
    """
    Represents a file attached to a chat message.
    """
    message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, related_name='files')
    file = models.FileField(upload_to=chat_attachment_path, max_length=500)
    original_filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50, help_text="MIME type or simple type like 'image', 'pdf', etc.")

    extracted_text = models.TextField(blank=True, null=True, help_text="Text content extracted from the file.")
    metadata = models.JSONField(default=dict, blank=True, help_text="Additional metadata about the file (e.g. video duration, zip structure).")

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"File {self.original_filename} for Message {self.message.id}"

    def delete(self, *args, **kwargs):
        storage = self.file.storage
        if storage and self.file.name and storage.exists(self.file.name):
            storage.delete(self.file.name)
        super().delete(*args, **kwargs)
