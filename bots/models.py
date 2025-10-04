# bots/models.py
from django.db import models
from django.conf import settings

class Category(models.Model):
    """Represents a category for bots."""
    name = models.CharField(max_length=100, unique=True)
    
    def __str__(self):
        return self.name

class Bot(models.Model):
    """Represents an AI bot created by a user."""
    class Publicity(models.TextChoices):
        PRIVATE = 'Private', 'Private'
        GUESTS = 'Guests', 'Guests'
        PUBLIC = 'Public', 'Public'

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='bots')
    name = models.CharField(max_length=100)
    prompt = models.TextField()
    avatar_url = models.URLField(max_length=2048, blank=True, null=True)
    voice = models.CharField(max_length=50, default='EnergeticYouth')
    language = models.CharField(max_length=50, default='English')
    publicity = models.CharField(max_length=10, choices=Publicity.choices, default=Publicity.PUBLIC)
    
    is_official = models.BooleanField(default=False) # Add this field
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True, related_name='bots') # Add this field
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name