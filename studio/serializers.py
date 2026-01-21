from rest_framework import serializers
from .models import KnowledgeArtifact, KnowledgeSource

class KnowledgeArtifactSerializer(serializers.ModelSerializer):
    # Write-only configuration fields
    quantity = serializers.IntegerField(write_only=True, required=False)
    difficulty = serializers.CharField(write_only=True, required=False)
    source_ids = serializers.ListField(child=serializers.CharField(), write_only=True, required=False)
    custom_instructions = serializers.CharField(write_only=True, required=False)
    # include_chat_history REMOVED
    
    # Input field for Podcast duration (Short/Medium/Long)
    duration = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = KnowledgeArtifact
        fields = [
            'id', 'chat', 'type', 'title', 'status',
            'content', 'media_url', 'score', 'created_at',
            # 'duration' is explicitly defined above as write_only
            'duration',
            # Write only config
            'quantity', 'difficulty', 'source_ids', 'custom_instructions'
        ]
        read_only_fields = ['id', 'created_at']
        extra_kwargs = {
            'content': {'required': False, 'allow_null': True},
            'media_url': {'required': False, 'allow_null': True},
            'score': {'required': False, 'allow_null': True},
            'status': {'read_only': True}
        }

    def create(self, validated_data):
        # Remove fields that are not part of the model
        validated_data.pop('quantity', None)
        validated_data.pop('difficulty', None)
        validated_data.pop('source_ids', None)
        validated_data.pop('custom_instructions', None)
        validated_data.pop('duration', None)

        return super().create(validated_data)

    def validate(self, data):
        """
        Validate content structure based on artifact type.
        """
        artifact_type = data.get('type')
        content = data.get('content')

        # SKIP validation if content is missing or null
        if not content:
            return data

        try:
            if artifact_type == KnowledgeArtifact.ArtifactType.QUIZ:
                self._validate_quiz(content)
            elif artifact_type == KnowledgeArtifact.ArtifactType.SLIDE:
                self._validate_slide(content)
            elif artifact_type == KnowledgeArtifact.ArtifactType.FLASHCARD:
                self._validate_flashcard(content)
        except Exception as e:
            # Catch unexpected validation logic errors and re-raise as ValidationError
            if isinstance(e, serializers.ValidationError):
                raise e
            raise serializers.ValidationError({"content": f"Invalid format: {str(e)}"})

        return data

    def _validate_quiz(self, content):
        if not isinstance(content, list):
            raise serializers.ValidationError({"content": "Quiz content must be a list of questions."})

        for idx, item in enumerate(content):
            if not isinstance(item, dict):
                raise serializers.ValidationError({"content": f"Item {idx} must be a dictionary."})

            required = ['question', 'options', 'correctAnswerIndex']
            for field in required:
                if field not in item:
                    raise serializers.ValidationError({"content": f"Item {idx} missing required field '{field}'."})

            if not isinstance(item['options'], list):
                raise serializers.ValidationError({"content": f"Item {idx} 'options' must be a list."})

    def _validate_slide(self, content):
        if not isinstance(content, list):
            raise serializers.ValidationError({"content": "Slide content must be a list of pages."})

        for idx, item in enumerate(content):
            if not isinstance(item, dict):
                raise serializers.ValidationError({"content": f"Item {idx} must be a dictionary."})

            required = ['title', 'bullets']
            for field in required:
                if field not in item:
                    raise serializers.ValidationError({"content": f"Item {idx} missing required field '{field}'."})

            if not isinstance(item['bullets'], list):
                raise serializers.ValidationError({"content": f"Item {idx} 'bullets' must be a list."})

    def _validate_flashcard(self, content):
        if not isinstance(content, list):
            raise serializers.ValidationError({"content": "Flashcard content must be a list of cards."})

        for idx, item in enumerate(content):
            if not isinstance(item, dict):
                raise serializers.ValidationError({"content": f"Item {idx} must be a dictionary."})

            required = ['front', 'back']
            for field in required:
                if field not in item:
                    raise serializers.ValidationError({"content": f"Item {idx} missing required field '{field}'."})

class KnowledgeSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeSource
        fields = [
            'id', 'user', 'title', 'source_type',
            'file', 'url', 'extracted_text', 'metadata',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at', 'user', 'extracted_text']
        extra_kwargs = {
            'file': {'required': False, 'allow_null': True},
            'url': {'required': False, 'allow_null': True}
        }
