# accounts/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

# Custom user extends AbstractUser so we can add 'is_email_verified'
class User(AbstractUser):
    # email should be unique for verification flows
    email = models.EmailField(unique=True)
    is_email_verified = models.BooleanField(default=False)

    def __str__(self):
        return self.username

