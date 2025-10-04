# bots/admin.py
from django.contrib import admin
from .models import Bot, Category

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    """
    Admin interface options for the Category model.
    """
    list_display = ('name', 'id')
    search_fields = ('name',)

@admin.register(Bot)
class BotAdmin(admin.ModelAdmin):
    """
    Admin interface options for the Bot model.
    """
    list_display = ('name', 'owner', 'category', 'publicity', 'is_official')
    list_filter = ('is_official', 'publicity', 'category')
    search_fields = ('name', 'owner__username')
    # This allows you to easily edit bot details in the admin panel
    fields = ('name', 'prompt', 'owner', 'category', 'publicity', 'is_official', 'avatar_url', 'voice', 'language')
    # Make owner field searchable instead of a dropdown for performance
    raw_id_fields = ('owner',)