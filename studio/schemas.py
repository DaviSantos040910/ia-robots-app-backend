# studio/schemas.py

from google.genai import types

# Esquema para Quiz
# {"type": "array", "items": {...}}
QUIZ_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "question": types.Schema(type=types.Type.STRING),
            "options": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING)
            ),
            "correctAnswerIndex": types.Schema(type=types.Type.INTEGER)
        },
        required=["question", "options", "correctAnswerIndex"]
    )
)

# Esquema para Flashcards
FLASHCARD_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "front": types.Schema(type=types.Type.STRING),
            "back": types.Schema(type=types.Type.STRING)
        },
        required=["front", "back"]
    )
)

# Esquema para Resumo
SUMMARY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "summary": types.Schema(type=types.Type.STRING),
        "key_points": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING)
        )
    },
    required=["summary"]
)

# Esquema para Slides (Adicional para completude, embora não explicitamente pedido, é útil dado o código existente)
SLIDE_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "title": types.Schema(type=types.Type.STRING),
            "bullets": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING)
            )
        },
        required=["title", "bullets"]
    )
)
