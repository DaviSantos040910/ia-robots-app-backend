from django.db import models
from chat.models import Chat

class KnowledgeArtifact(models.Model):
    class ArtifactType(models.TextChoices):
        SLIDE = 'SLIDE', 'Slide'
        QUIZ = 'QUIZ', 'Quiz'
        FLASHCARD = 'FLASHCARD', 'Flashcard'
        SPREADSHEET = 'SPREADSHEET', 'Spreadsheet'
        WORKBOOK = 'WORKBOOK', 'Workbook'
        PODCAST = 'PODCAST', 'Podcast'
        SUMMARY = 'SUMMARY', 'Summary'

    class Status(models.TextChoices):
        PROCESSING = 'processing', 'Processing'
        READY = 'ready', 'Ready'
        ERROR = 'error', 'Error'

    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='artifacts')
    type = models.CharField(max_length=20, choices=ArtifactType.choices)
    title = models.CharField(max_length=255)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PROCESSING
    )

    # Polymorphic content (JSON structure differs by type)
    content = models.JSONField(null=True, blank=True)

    # Specific fields for Podcast or other media
    media_url = models.URLField(null=True, blank=True)
    duration = models.CharField(max_length=20, null=True, blank=True) # "5:30", "10:00"

    # Optional score if we want to store user performance on a quiz artifact here?
    # The frontend interface shows `score?: string`.
    score = models.CharField(max_length=20, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.type}: {self.title} ({self.status})"
