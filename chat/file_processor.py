import os
import logging
import zipfile
import io
import pypdf
import docx
from typing import List, Dict, Any
from django.core.files.uploadedfile import UploadedFile
from youtube_transcript_api import YouTubeTranscriptApi
from .services.transcription_service import transcribe_audio_gemini

logger = logging.getLogger(__name__)

class FileProcessor:
    @staticmethod
    def process_file(file: UploadedFile, file_type: str = None) -> Dict[str, Any]:
        """
        Processa o arquivo e retorna o texto extraído e metadados.
        Retorna:
        {
            'extracted_text': str,
            'metadata': dict
        }
        """
        logger.info(f"Processing file: {file.name}, type: {file_type}")

        extracted_text = ""
        metadata = {}

        try:
            # 1. YouTube Link (in text file)
            # Verifica se é arquivo texto e se o conteúdo é apenas um link do YouTube
            if file.name.lower().endswith('.txt'):
                 # Lê conteúdo para verificação
                content = file.read().decode('utf-8').strip()
                file.seek(0) # Reset pointer

                if "youtube.com/watch" in content or "youtu.be/" in content:
                    # É provavelmente um link do youtube
                    lines = content.splitlines()
                    # Simples check: se tiver poucas linhas e parecer URL
                    if len(lines) < 2 and any(x in content for x in ["youtube.com", "youtu.be"]):
                         # Processar como YouTube
                         video_id = FileProcessor._extract_youtube_id(content)
                         if video_id:
                             transcript = FileProcessor._get_youtube_transcript(video_id)
                             extracted_text = transcript
                             metadata['source'] = 'youtube_transcript'
                             metadata['video_id'] = video_id
                             return {'extracted_text': extracted_text, 'metadata': metadata}

            # 2. ZIP Processing
            if file.name.lower().endswith('.zip') or file_type == 'application/zip':
                return FileProcessor._process_zip(file)

            # 3. Audio Processing
            # Check mime type or extension
            if file_type == 'audio' or file.name.lower().endswith(('.mp3', '.m4a', '.wav', '.ogg')):
                 return FileProcessor._process_audio(file)

            # 4. Standard Text/PDF/DOCX
            extracted_text = FileProcessor.extract_text(file)

            return {
                'extracted_text': extracted_text,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            return {
                'extracted_text': "",
                'metadata': {'error': str(e)}
            }

    @staticmethod
    def _process_zip(file: UploadedFile) -> Dict[str, Any]:
        extracted_text = ""
        structure = []

        try:
            with zipfile.ZipFile(file) as z:
                # Listar estrutura
                structure = z.namelist()

                for filename in structure:
                    # Ignorar diretórios e arquivos ocultos/sistemas
                    if filename.endswith('/') or filename.startswith('__MACOSX') or filename.startswith('.'):
                        continue

                    # Filtrar extensões de código/texto relevantes
                    if filename.lower().endswith(('.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.md', '.txt', '.json', '.csv')):
                        with z.open(filename) as f:
                            try:
                                content = f.read().decode('utf-8', errors='replace')
                                extracted_text += f"\n\n--- FILE: {filename} ---\n\n"
                                extracted_text += content
                            except Exception as read_err:
                                logger.warning(f"Could not read file {filename} in zip: {read_err}")

        except zipfile.BadZipFile:
            logger.error("Invalid zip file")
            return {'extracted_text': '', 'metadata': {'error': 'Bad Zip File'}}

        return {
            'extracted_text': extracted_text,
            'metadata': {'structure': structure, 'type': 'zip_content'}
        }

    @staticmethod
    def _process_audio(file: UploadedFile) -> Dict[str, Any]:
        result = transcribe_audio_gemini(file)
        if result.get('success'):
            return {
                'extracted_text': result.get('transcription', ''),
                'metadata': {'type': 'audio_transcription'}
            }
        else:
            return {
                'extracted_text': '',
                'metadata': {'error': result.get('error'), 'type': 'audio_transcription_failed'}
            }

    @staticmethod
    def _extract_youtube_id(url: str) -> str:
        """
        Extracts video ID from YouTube URL.
        """
        import urllib.parse

        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                p = urllib.parse.parse_qs(parsed_url.query)
                return p['v'][0]
            if parsed_url.path[:7] == '/embed/':
                return parsed_url.path.split('/')[2]
            if parsed_url.path[:3] == '/v/':
                return parsed_url.path.split('/')[2]
        return None

    @staticmethod
    def _get_youtube_transcript(video_id: str) -> str:
        try:
            # Check if YouTubeTranscriptApi has list_transcripts (standard 0.6.x) or list (1.2.3)
            # Based on inspection, this env has version 1.2.3 with list() method on instance

            api = YouTubeTranscriptApi()

            # Defensive check for different versions/APIs
            if hasattr(api, 'list'):
                transcript_list = api.list(video_id)
            elif hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                 # Fallback to static method if API changes (unlikely given pip show)
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            elif hasattr(YouTubeTranscriptApi, 'get_transcript'):
                 # Fallback to old static shortcut
                 transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'en'])
                 # This usually returns list of dicts directly
                 return FileProcessor._format_transcript_data(transcript)
            else:
                return "YouTube API incompatible or method not found."

            # If we got a transcript list object
            if transcript_list:
                # Tenta encontrar PT ou EN
                transcript = transcript_list.find_transcript(['pt', 'en'])
                fetched_transcript = transcript.fetch()
                return FileProcessor._format_transcript_data(fetched_transcript)

            return "No transcript found."

        except Exception as e:
            logger.error(f"Error fetching YouTube transcript for {video_id}: {e}")
            return f"Error fetching transcript: {e}"

    @staticmethod
    def _format_transcript_data(data) -> str:
        """
        Helper to format transcript data whether it is list of dicts or objects.
        """
        text_lines = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text_lines.append(item.get('text', ''))
                elif hasattr(item, 'text'):
                    text_lines.append(item.text)
                else:
                    text_lines.append(str(item))
        else:
             # Single object?
             if hasattr(data, 'text'):
                 text_lines.append(data.text)
             else:
                 text_lines.append(str(data))

        return "\n".join(text_lines)

    @staticmethod
    def extract_text(file, mime_type: str = None) -> str:
        """
        Extrai texto de arquivos PDF, DOCX ou TXT de forma robusta.
        Aceita tanto path (str) quanto file-like object (UploadedFile).
        """
        # Se for UploadedFile ou file-like, usar a interface apropriada
        # Mas as libs pypdf e docx preferem caminhos ou file-like objects seekable

        file_path = None
        if isinstance(file, str):
            file_path = file
            if not os.path.exists(file_path):
                 return ""
            file_name = file_path
        else:
             # Assumindo UploadedFile ou objeto similar do Django
             file_name = getattr(file, 'name', '')
        
        text = ""
        try:
            # Tenta inferir mime_type pela extensão se não fornecido
            if not mime_type:
                if file_name.lower().endswith('.pdf'):
                    mime_type = 'application/pdf'
                elif file_name.lower().endswith('.docx'):
                    mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                elif file_name.lower().endswith('.txt'):
                    mime_type = 'text/plain'

            # --- Processamento PDF ---
            if mime_type == 'application/pdf':
                try:
                    if file_path:
                        reader = pypdf.PdfReader(file_path)
                    else:
                        reader = pypdf.PdfReader(file)

                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                except Exception as e:
                    logger.error(f"Erro ao ler PDF: {e}")

            # --- Processamento DOCX ---
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                try:
                    if file_path:
                         doc = docx.Document(file_path)
                    else:
                         doc = docx.Document(file)

                    text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
                except Exception as e:
                    logger.error(f"Erro ao ler DOCX: {e}")

            # --- Processamento Texto Puro ---
            elif mime_type and mime_type.startswith('text/') or file_name.lower().endswith('.txt'):
                try:
                    if file_path:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                    else:
                         # UploadedFile
                         file.seek(0)
                         text = file.read().decode('utf-8', errors='ignore')
                except Exception as e:
                    logger.error(f"Erro ao ler TXT: {e}")
            
            else:
                logger.warning(f"Tipo de arquivo não suportado para extração: {mime_type}")

        except Exception as e:
            logger.error(f"Erro genérico no FileProcessor: {e}")
            return ""
        
        # Limpeza básica: remove excesso de espaços em branco e quebras de linha múltiplas
        return " ".join(text.split())

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Divide o texto em blocos menores com sobreposição (overlap) para manter contexto.
        """
        if not text: return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            
            # Tenta não cortar palavras no meio: procura o último espaço antes do corte
            if end < text_len:
                last_space = text.rfind(' ', max(start, end - 100), end)
                if last_space != -1:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Se atingiu o final, para
            if end >= text_len:
                break
                
            # Avança o cursor, recuando pelo tamanho do overlap para garantir continuidade
            start = end - overlap
            
        return chunks
