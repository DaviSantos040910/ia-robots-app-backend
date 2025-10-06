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
    
    # --- CORREÇÃO: A CLASSE 'Publicity' FOI RESTAURADA AQUI ---
    # This class defines the choices for the 'publicity' field.
    # It needs to be defined inside the Bot model before it is used.
    class Publicity(models.TextChoices):
        PRIVATE = 'Private', 'Private'
        GUESTS = 'Guests', 'Guests'
        PUBLIC = 'Public', 'Public'

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='bots')
    name = models.CharField(max_length=100)
    prompt = models.TextField()
    avatar_url = models.URLField(max_length=2048, blank=True, null=True)
    voice = models.CharField(max_length=50, default='EnergeticYouth')
    
    # This line now works because 'Publicity' is defined above.
    publicity = models.CharField(
        max_length=10, 
        choices=Publicity.choices, 
        default=Publicity.PUBLIC
    )
    
    is_official = models.BooleanField(default=False)
    
    # Switched from ForeignKey to ManyToManyField
    categories = models.ManyToManyField(Category, related_name='bots', blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name