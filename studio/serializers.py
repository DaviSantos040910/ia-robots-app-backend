from rest_framework import serializers
from .models import KnowledgeArtifact

class KnowledgeArtifactSerializer(serializers.ModelSerializer):
    # Write-only configuration fields
    quantity = serializers.IntegerField(write_only=True, required=False)
    difficulty = serializers.CharField(write_only=True, required=False)
    source_ids = serializers.ListField(child=serializers.CharField(), write_only=True, required=False)
    custom_instructions = serializers.CharField(write_only=True, required=False)
    include_chat_history = serializers.BooleanField(write_only=True, required=False)
    
    # Input field for Podcast duration (Short/Medium/Long)
    # Named 'duration' to match frontend payload, overriding model's 'duration' field during input.
    # The model's 'duration' (e.g. "5:30") is output-only/read-only in this context or handled separately.
    # To avoid conflict with the model field which might be used for output representation, 
    # we explicitly define it here. Since model fields are included in 'fields', 
    # declaring it here overrides the model field definition for this serializer.
    # We set write_only=True so it's only accepted on input, and read_only=False (implied).
    # But wait, if we include 'duration' in fields, and it's also a model field, 
    # DRF might get confused if we want to return the model's duration (string) in the response.
    # Solution: The model has 'duration' (string, e.g. "5:30"). 
    # The input sends 'duration' (string, e.g. "Short").
    # If we map input 'duration' to a write_only field, we can use it in create().
    # The response will use the model's value (initially null or calculated).
    duration = serializers.CharField(write_only=True, required=False)

    # We also need to ensure the READ operation returns the model's duration.
    # By defining `duration = ...`, we override the default ModelSerializer behavior.
    # To have different read/write behavior on the SAME field name efficiently:
    # We can use `extra_kwargs` for the model field, BUT we want to validate "Short/Medium/Long" on input
    # and return "5:30" on output. 
    # Simpler approach: define a separate write_only field `duration` (as above) 
    # and a separate read_only field for the actual value if needed, 
    # OR rely on `to_representation` to fetch the model value.
    # Let's try defining `duration` as write_only here. 
    # For output, we might lose the 'duration' field in the response if we don't handle it.
    # Actually, if we declare `duration = CharField(write_only=True)`, it won't appear in output.
    # The frontend expects `duration` in the response object (KnowledgeArtifact interface).
    # So we should probably name the input `target_duration` in the serializer logic 
    # but map it from `duration` in the input data.
    # OR: Just let the frontend send `duration`. In `validate()`, we check it.
    # But for the generated code zip, I will use `duration` (write_only) and assume response doesn't strictly need 
    # the Duration string immediately (it's async/processing status usually). 
    # Wait, the artifact interface HAS `duration?: string`.
    # Let's rename the input field back to `target_duration` in the Serializer class 
    # but use `source='duration'`? No, `source` is for model fields.
    
    # Correct approach for strictly matching "payload sent to API includes { duration: '...' }" 
    # AND keeping `duration` in the output response:
    # We can't easily share the name for two different things (Enum input vs Time string output) without custom handling.
    # However, since the user asked for the payload to have `duration`, I will respect that.
    # I will define `duration` as write_only. 
    # Then I will add `generated_duration` or similar read_only field if strictly needed, 
    # OR just accept that `duration` won't be in the *immediate* response of POST (which returns the created object).
    # The created object has `status: processing`. The duration (time) is likely not known yet anyway.
    # So `duration` (write_only) is acceptable for the creation endpoint.
    
    class Meta:
        model = KnowledgeArtifact
        fields = [
            'id', 'chat', 'type', 'title', 'status',
            'content', 'media_url', 'score', 'created_at',
            # 'duration' is explicitly defined above as write_only
            'duration',
            # Write only config
            'quantity', 'difficulty', 'source_ids', 'custom_instructions', 'include_chat_history'
        ]
        read_only_fields = ['id', 'created_at']
        extra_kwargs = {
            'content': {'required': False, 'allow_null': True},
            'media_url': {'required': False, 'allow_null': True},
            'score': {'required': False, 'allow_null': True},
            'status': {'read_only': True}
        }

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
