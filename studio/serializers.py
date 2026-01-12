from rest_framework import serializers
from .models import KnowledgeArtifact

class KnowledgeArtifactSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeArtifact
        fields = [
            'id', 'chat', 'type', 'title', 'status',
            'content', 'media_url', 'duration', 'score', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']

    def validate(self, data):
        """
        Validate content structure based on artifact type.
        """
        artifact_type = data.get('type')
        content = data.get('content')

        # If we are just updating status or other fields and content is not present, skip validation
        if not content:
            return data

        if artifact_type == KnowledgeArtifact.ArtifactType.QUIZ:
            self._validate_quiz(content)
        elif artifact_type == KnowledgeArtifact.ArtifactType.SLIDE:
            self._validate_slide(content)
        elif artifact_type == KnowledgeArtifact.ArtifactType.FLASHCARD:
            self._validate_flashcard(content)

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
