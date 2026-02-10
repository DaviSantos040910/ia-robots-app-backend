import logging
import json
import time
from django.utils import timezone
from django_rq import job
from rq import Retry

from studio.models import KnowledgeArtifact
from studio.services.source_assembler import SourceAssemblyService
from studio.services.podcast_scripting import PodcastScriptingService
from studio.services.audio_mixer import AudioMixerService
from chat.services.ai_client import get_ai_client
from studio.schemas import QUIZ_SCHEMA, FLASHCARD_SCHEMA, SUMMARY_SCHEMA, SLIDE_SCHEMA
from google.genai import types
from core.genai_models import GENAI_MODEL_TEXT

logger = logging.getLogger(__name__)

@job('default', timeout=360, result_ttl=86400, retry=Retry(max=3))
def generate_artifact_job(artifact_id, options):
    start_time = time.time()
    try:
        artifact = KnowledgeArtifact.objects.get(id=artifact_id)
    except KnowledgeArtifact.DoesNotExist:
        logger.error(f"Artifact {artifact_id} not found.")
        return

    # Update Start State
    artifact.stage = KnowledgeArtifact.Stage.ASSEMBLING_CONTEXT
    artifact.started_at = timezone.now()
    artifact.attempts += 1
    artifact.save(update_fields=['stage', 'started_at', 'attempts'])

    try:
        # 1. ASSEMBLING CONTEXT
        config = {
            'selectedSourceIds': options.get('source_ids', []),
            'includeChatHistory': options.get('includeChatHistory', False)
        }
        full_context = SourceAssemblyService.get_context_from_config(
            artifact.chat.id,
            config,
            query=artifact.title
        )

        # 2. GENERATING CONTENT
        artifact.stage = KnowledgeArtifact.Stage.GENERATING
        artifact.save(update_fields=['stage'])

        if artifact.type == KnowledgeArtifact.ArtifactType.PODCAST:
            _generate_podcast(artifact, full_context, options)
        else:
            _generate_standard_artifact(artifact, full_context, options)

        # 3. READY
        artifact.stage = KnowledgeArtifact.Stage.READY
        artifact.status = KnowledgeArtifact.Status.READY
        artifact.finished_at = timezone.now()
        artifact.save(update_fields=['stage', 'status', 'finished_at', 'media_url', 'duration', 'content'])

        duration = (time.time() - start_time) * 1000
        logger.info(f"[{artifact.correlation_id}] Artifact {artifact_id} generated successfully in {duration:.2f}ms")

    except Exception as e:
        logger.error(f"[{artifact.correlation_id}] Job failed for artifact {artifact_id}: {e}", exc_info=True)
        artifact.stage = KnowledgeArtifact.Stage.ERROR
        artifact.status = KnowledgeArtifact.Status.ERROR
        artifact.error_message = str(e)
        artifact.finished_at = timezone.now()
        artifact.save(update_fields=['stage', 'status', 'error_message', 'finished_at'])
        raise e # Re-raise to trigger RQ retry if configured

def _generate_podcast(artifact, context, options):
    # This logic was extracted from View
    # 1. Generate Script
    script = PodcastScriptingService.generate_script(
        title=artifact.title,
        context=context,
        duration_constraint=options.get('target_duration', 'Medium')
    )
    artifact.content = script
    # We update content here so if mixing fails, we at least have the script?
    # Or just wait until end. The main job logic saves content at the end.
    # But podcast helper updates artifact object instance.

    # 2. Rendering Audio (Mixing)
    artifact.stage = KnowledgeArtifact.Stage.RENDERING_EXPORT
    artifact.save(update_fields=['stage'])

    audio_path = AudioMixerService.mix_podcast(script)

    artifact.media_url = f"/media/{audio_path}"
    artifact.duration = options.get('target_duration', '10:00')

def _generate_standard_artifact(artifact, full_context, options):
    client = get_ai_client()
    # model_name = get_model('chat') # Already updated to use GENAI_MODEL_TEXT
    model_name = GENAI_MODEL_TEXT

    bot_prompt = artifact.chat.bot.prompt if artifact.chat.bot and artifact.chat.bot.prompt else None
    system_instruction, response_schema = _build_prompt_and_schema(
        artifact.type,
        artifact.title,
        full_context,
        options,
        bot_prompt=bot_prompt
    )

    temperature = 0.3 if options.get('source_ids') else 0.7

    generate_config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
        response_mime_type="application/json"
    )

    if response_schema:
        generate_config.response_schema = response_schema

    response = client.models.generate_content(
        model=model_name,
        contents="Generate the artifact content based on the system instructions and context.",
        config=generate_config
    )

    if response.parsed:
        artifact.content = response.parsed
    else:
        try:
            artifact.content = json.loads(response.text)
        except:
             if artifact.type == KnowledgeArtifact.ArtifactType.SUMMARY:
                 artifact.content = {"summary": response.text}
             else:
                 raise ValueError("Failed to parse JSON response")

def _build_prompt_and_schema(artifact_type, title, context, options, bot_prompt=None):
    difficulty = options.get('difficulty', 'Medium')
    quantity = options.get('quantity', 10)
    instructions = options.get('custom_instructions', '')

    base_instruction = (
        f"You are an expert educational content generator. "
        f"Create a {artifact_type} titled '{title}'.\n"
        f"Language: Detect the language from the context (default to Portuguese if unclear).\n"
        f"Target Audience Difficulty: {difficulty}.\n"
    )

    if bot_prompt:
            base_instruction += f"\nYOUR PERSONALITY/ROLE:\n{bot_prompt}\n"
            base_instruction += "Adopt this persona for the tone and style of the content, but STRICTLY use the provided Context Material for facts.\n"

    if instructions:
        base_instruction += f"CUSTOM INSTRUCTIONS:\n{instructions}\n"

    base_instruction += f"\nCONTEXT MATERIAL (Source Files Only):\n{context}\n"

    schema = None

    if artifact_type == KnowledgeArtifact.ArtifactType.QUIZ:
        base_instruction += f"Generate exactly {quantity} questions."
        schema = QUIZ_SCHEMA

    elif artifact_type == KnowledgeArtifact.ArtifactType.FLASHCARD:
        base_instruction += f"Generate exactly {quantity} cards."
        schema = FLASHCARD_SCHEMA

    elif artifact_type == KnowledgeArtifact.ArtifactType.SUMMARY:
        base_instruction += "Generate a comprehensive summary and key points."
        schema = SUMMARY_SCHEMA

    elif artifact_type == KnowledgeArtifact.ArtifactType.SLIDE:
        base_instruction += f"Generate exactly {quantity} slides."
        schema = SLIDE_SCHEMA

    return base_instruction, schema
