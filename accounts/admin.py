# accounts/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from .models import User

@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    # Display fields in admin
    list_display = ("username", "email", "is_email_verified", "is_staff", "is_superuser")
    list_filter = ("is_email_verified", "is_staff", "is_superuser")
    search_fields = ("username", "email")
