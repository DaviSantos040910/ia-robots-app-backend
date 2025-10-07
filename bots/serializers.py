# bots/serializers.py
from rest_framework import serializers
from .models import Bot, Category

class CategorySerializer(serializers.ModelSerializer):
    """
    Serializer for the Category model.
    """
    class Meta:
        model = Category
        fields = ('id', 'name', 'translation_key')

class BotSerializer(serializers.ModelSerializer):
    """
    General purpose serializer for the Bot model.
    """
    categories = CategorySerializer(many=True, read_only=True)
    category_ids = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(), source='categories', write_only=True, many=True, required=False
    )
    owner_username = serializers.ReadOnlyField(source='owner.username')

    class Meta:
        model = Bot
        fields = (
            'id', 'name', 'description','prompt', 'avatar_url', 'voice',
            'publicity', 'is_official', 'owner', 'owner_username',
            'categories', 'category_ids'
        )
        read_only_fields = ('owner',)

    def validate_category_ids(self, value):
        if len(value) > 3:
            raise serializers.ValidationError("You can select a maximum of 3 categories.")
        return value

class BotDetailSerializer(serializers.ModelSerializer):
    """
    Detailed serializer specifically for the Bot Settings screen.
    """
    stats = serializers.SerializerMethodField()
    tags = serializers.SerializerMethodField()
    createdByMe = serializers.SerializerMethodField()
    settings = serializers.SerializerMethodField()
    handle = serializers.ReadOnlyField(source='owner.username')

    class Meta:
        model = Bot
        fields = (
            'id', 'name', 'handle', 'avatar_url', 'stats', 'tags', 
            'createdByMe', 'settings'
        )

    def get_stats(self, obj):
        return { "monthlyUsers": "0", "followers": "0" }

    def get_tags(self, obj):
        tags = []
        if obj.is_official:
            tags.append('official')
        for cat in obj.categories.all():
            tags.append(cat.name.lower())
        return tags

    def get_createdByMe(self, obj):
        request = self.context.get('request')
        if request and hasattr(request, 'user'):
            return obj.owner == request.user
        return False

    def get_settings(self, obj):
        return {
            "voice": obj.voice,
            "publicity": obj.publicity
        }