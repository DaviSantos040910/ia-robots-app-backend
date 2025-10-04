# accounts/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    email = models.EmailField(unique=True)
    is_email_verified = models.BooleanField(default=False)
    is_premium = models.BooleanField(default=False) # Add this line

    def __str__(self):
        return self.username