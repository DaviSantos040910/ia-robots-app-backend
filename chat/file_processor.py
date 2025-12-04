import os
import logging
from typing import List
import pypdf
import docx

logger = logging.getLogger(__name__)

class FileProcessor:
    @staticmethod
    def extract_text(file_path: str, mime_type: str) -> str:
        if not os.path.exists(file_path): return ""
        text = ""
        try:
            if mime_type == 'application/pdf':
                reader = pypdf.PdfReader(file_path)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                doc = docx.Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            elif mime_type.startswith('text/'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
        except Exception as e:
            logger.error(f"Erro ao ler arquivo: {e}")
            return ""
        return " ".join(text.split())

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        if not text: return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Tenta não cortar palavras no final
            if end < text_len:
                last_space = text.rfind(' ', max(start, end - 100), end)
                if last_space != -1: end = last_space
            chunk = text[start:end].strip()
            if chunk: chunks.append(chunk)
            if end == text_len: break
            start = end - overlap # Overlap para manter contexto
        return chunks