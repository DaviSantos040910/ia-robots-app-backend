# bots/serializers.py
from rest_framework import serializers
from .models import Bot, Category

class CategorySerializer(serializers.ModelSerializer):
    """
    Serializer for the Category model.
    """
    class Meta:
        model = Category
        fields = ('id', 'name')

class BotSerializer(serializers.ModelSerializer):
    """
    General purpose serializer for the Bot model.
    Used for creating, listing, and updating bots, including for admin purposes.
    """
    # Use the CategorySerializer for read operations to show category details.
    category = CategorySerializer(read_only=True)
    
    # Use PrimaryKeyRelatedField for write operations (create/update).
    # This allows assigning a category by sending its ID.
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(), source='category', write_only=True, required=False, allow_null=True
    )
    
    # Make owner's username readable in lists
    owner_username = serializers.ReadOnlyField(source='owner.username')


    class Meta:
        model = Bot
        fields = (
            'id', 'name', 'prompt', 'avatar_url', 'voice',
            'language', 'publicity', 'is_official', 'owner',
            'owner_username', 'category', 'category_id'
        )
        # The owner is automatically set to the logged-in user upon creation.
        read_only_fields = ('owner',)

class BotDetailSerializer(serializers.ModelSerializer):
    """
    Detailed serializer specifically for the Bot Settings screen in the frontend.
    It structures the data (stats, tags, settings) exactly as the mobile app expects.
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
        """
        Returns placeholder stats. In a real application, this could be calculated
        based on usage data.
        """
        # TODO: Implement real stats calculation
        return {
            "monthlyUsers": "0",
            "followers": "0"
        }

    def get_tags(self, obj):
        """
        Returns the bot's category name as a tag, plus an 'official' tag if applicable.
        """
        tags = []
        if obj.is_official:
            tags.append('official')
        if obj.category:
            tags.append(obj.category.name.lower())
        return tags

    def get_createdByMe(self, obj):
        """
        Checks if the user making the request is the owner of the bot.
        """
        request = self.context.get('request')
        if request and hasattr(request, 'user'):
            return obj.owner == request.user
        return False

    def get_settings(self, obj):
        """
        Nests bot settings into a 'settings' object as the frontend expects.
        """
        return {
            "voice": obj.voice,
            "language": obj.language,
            "publicity": obj.publicity
        }