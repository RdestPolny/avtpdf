"""
Redaktor AI - Interaktywny Procesor Dokumentów z PaddleOCR
===========================================================

WDROŻENIE NA STREAMLIT CLOUD:
-----------------------------
W repozytorium muszą znajdować się pliki:
1. `requirements.txt` (z listą bibliotek Python, najlepiej z przypiętymi wersjami)
2. `packages.txt` (z listą bibliotek systemowych, np. libgl1-mesa-glx)

URUCHOMIENIE LOKALNE:
--------------------
streamlit run redaktor_ai_enhanced.py
"""

import streamlit as st
import fitz  # PyMuPDF
from openai import AsyncOpenAI
import io
import zipfile
import json
from pathlib import Path
import re
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Opcjonalne importy
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    MAMMOTH_AVAILABLE = False

# Importy PaddleOCR
PADDLEOCR_AVAILABLE = False
try:
    import numpy as np
    from PIL import Image
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass

# ===== KONFIGURACJA =====

PROJECTS_DIR = Path("pdf_processor_projects")
BATCH_SIZE = 10
MAX_RETRIES = 3
DEFAULT_MODEL = 'gpt-4o-mini'
OCR_CONFIDENCE_THRESHOLD = 0.6
NATIVE_TEXT_MIN_LENGTH = 50

SESSION_STATE_DEFAULTS = {
    'processing_status': 'idle', 'document': None, 'current_page': 0,
    'total_pages': 0, 'extracted_pages': [], 'project_name': None,
    'next_batch_start_index': 0, 'uploaded_filename': None, 'api_key': None,
    'model': DEFAULT_MODEL, 'meta_tags': {}, 'project_loaded_and_waiting_for_file': False,
    'processing_mode': 'all', 'start_page': 1, 'end_page': 1,
    'processing_end_page_index': 0, 'article_page_groups_input': '',
    'article_groups': [], 'next_article_index': 0, 'file_type': None,
    'ocr_mode': 'paddleocr', 'ocr_language': 'pl', 'optimized_articles': {}
}

# ===== KLASY POMOCNICZE =====

@dataclass
class PageContent:
    page_number: int; text: str; images: List[Dict] = None
    extraction_method: str = "native"; ocr_confidence: float = 0.0
    def __post_init__(self):
        if self.images is None: self.images = []

def get_ocr_engine(language: str = 'pl'):
    """
    Pobiera lub tworzy instancję silnika PaddleOCR, bezpiecznie przechowując ją w st.session_state.
    """
    session_key = f"paddleocr_engine_{language}"
    if session_key in st.session_state and st.session_state[session_key] is not None:
        return st.session_state[session_key]
    if not PADDLEOCR_AVAILABLE:
        st.error("Biblioteka PaddleOCR nie jest zainstalowana!")
        return None
    try:
        from paddleocr import PaddleOCR
        with st.spinner(f"Inicjalizacja silnika OCR dla języka '{language}'..."):
            # === OSTATECZNA POPRAWKA ===
            # Usunięto wszystkie opcjonalne argumenty (`show_log`, `use_gpu`),
            # aby zapewnić kompatybilność z różnymi wersjami biblioteki.
            ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=language
            )
        st.session_state[session_key] = ocr_engine
        st.toast(f"✅ Silnik OCR gotowy!", icon="🚀")
        return ocr_engine
    except Exception as e:
        st.error(f"Krytyczny błąd inicjalizacji PaddleOCR: {e}")
        st.info("💡 Upewnij się, że plik 'packages.txt' istnieje, a wersje w 'requirements.txt' są poprawne.")
        return None

def extract_text_with_paddleocr(image_data: bytes, language: str = 'pl') -> Tuple[str, float]:
    """
    Wyciąga tekst z obrazu używając PaddleOCR.
    """
    try:
        ocr = get_ocr_engine(language)
        if ocr is None:
            st.warning("Silnik OCR nie jest dostępny z powodu błędu. Ekstrakcja pominięta.")
            return "", 0.0
        img = Image.open(io.BytesIO(image_data)); img_array = np.array(img)
        result = ocr.ocr(img_array, cls=True)
        if not result or not result[0]: return "", 0.0
        texts, confidences = [], []
        for line in result[0]:
            if line:
                texts.append(line[1][0]); confidences.append(line[1][1])
        full_text = "\n".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return full_text, avg_confidence
    except Exception as e:
        st.warning(f"Błąd podczas przetwarzania obrazu przez PaddleOCR: {e}")
        return "", 0.0

class DocumentHandler:
    def __init__(self, file_bytes: bytes, filename: str):
        self.file_bytes = file_bytes; self.filename = filename
        self.file_type = self._detect_file_type(filename)
        self._document = None; self._html_content = None; self._load_document()
    def _detect_file_type(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext == '.pdf': return 'pdf'
        elif ext == '.docx':
            if not DOCX_AVAILABLE: raise ValueError("Zainstaluj: pip install python-docx")
            return 'docx'
        elif ext == '.doc':
            if not MAMMOTH_AVAILABLE: raise ValueError("Zainstaluj: pip install mammoth")
            return 'doc'
        else: raise ValueError(f"Nieobsługiwany format pliku: {ext}")
    def _load_document(self):
        if self.file_type == 'pdf': self._document = fitz.open(stream=self.file_bytes, filetype="pdf")
        elif self.file_type == 'docx': self._document = DocxDocument(io.BytesIO(self.file_bytes))
        elif self.file_type == 'doc':
            result = mammoth.convert_to_html(io.BytesIO(self.file_bytes)); self._html_content = result.value
    def get_page_count(self) -> int:
        if self.file_type == 'pdf': return len(self._document)
        elif self.file_type == 'docx':
            all_text = '\n\n'.join([p.text for p in self._document.paragraphs]); words = all_text.split()
            return max(1, len(words) // 500 + (1 if len(words) % 500 > 0 else 0))
        elif self.file_type == 'doc':
            words = self._html_content.split()
            return max(1, len(words) // 500 + (1 if len(words) % 500 > 0 else 0))
        return 0
    def _should_use_ocr_primary(self, page_index: int) -> bool:
        if not PADDLEOCR_AVAILABLE: return False
        ocr_mode = st.session_state.get('ocr_mode', 'paddleocr')
        if ocr_mode == 'native': return False
        if ocr_mode == 'paddleocr': return True
        if self.file_type != 'pdf': return False # 'auto' mode
        try:
            page = self._document.load_page(page_index); native_text = page.get_text("text")
            if len(native_text.strip()) < NATIVE_TEXT_MIN_LENGTH: return True
            if not page.get_text("blocks"): return True
            return False
        except: return True
    def get_page_content(self, page_index: int, force_mode: str = None) -> PageContent:
        if self.file_type != 'pdf': return self._get_non_pdf_content(page_index)
        page = self._document.load_page(page_index); images = self._extract_images_from_pdf_page(page_index)
        use_ocr = force_mode == 'paddleocr' if force_mode else self._should_use_ocr_primary(page_index)
        if not use_ocr or not PADDLEOCR_AVAILABLE:
            return PageContent(page_index + 1, page.get_text("text"), images, "native")
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0)); img_bytes = pix.tobytes("png")
            language = st.session_state.get('ocr_language', 'pl')
            ocr_text, confidence = extract_text_with_paddleocr(img_bytes, language)
            if confidence > OCR_CONFIDENCE_THRESHOLD or len(ocr_text.strip()) > 50:
                return PageContent(page_index + 1, ocr_text, images, "paddleocr", confidence)
            else:
                native_text = page.get_text("text")
                if len(native_text.strip()) > len(ocr_text.strip()):
                    return PageContent(page_index + 1, native_text, images, "hybrid_native", confidence)
                else:
                    return PageContent(page_index + 1, ocr_text, images, "paddleocr_low_conf", confidence)
        except Exception as e:
            st.warning(f"OCR nie powiodło się dla strony {page_index + 1}: {e}. Używam metody natywnej.")
            return PageContent(page_index + 1, page.get_text("text"), images, "native_fallback")
    def _get_non_pdf_content(self, page_index: int) -> PageContent:
        if self.file_type == 'docx': return self._get_docx_page_content(page_index)
        elif self.file_type == 'doc': return self._get_doc_page_content(page_index)
    def _get_docx_page_content(self, page_index: int) -> PageContent:
        all_text = '\n\n'.join([p.text for p in self._document.paragraphs]); words = all_text.split()
        start = page_index * 500; end = min(start + 500, len(words))
        return PageContent(page_index + 1, ' '.join(words[start:end]), self._extract_images_from_docx())
    def _get_doc_page_content(self, page_index: int) -> PageContent:
        text = re.sub('<[^<]+?>', '', self._html_content); words = text.split()
        start = page_index * 500; end = min(start + 500, len(words))
        return PageContent(page_index + 1, ' '.join(words[start:end]), [])
    def _extract_images_from_pdf_page(self, page_index: int) -> List[Dict]:
        images = [];
        if self.file_type != 'pdf': return images
        try:
            page = self._document.load_page(page_index)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]; base_image = self._document.extract_image(xref)
                if base_image and base_image.get("width", 0) > 100 and base_image.get("height", 0) > 100:
                    images.append({'image': base_image['image'], 'ext': base_image['ext'], 'index': img_index})
        except Exception as e: st.warning(f"Nie udało się wyekstraktować obrazów ze strony {page_index + 1}: {e}")
        return images
    def _extract_images_from_docx(self) -> List[Dict]:
        images = []
        try:
            for rel in self._document.part.rels.values():
                if "image" in rel.target_ref:
                    img_data = rel.target_part.blob; ext = rel.target_ref.split('.')[-1]
                    images.append({'image': img_data, 'ext': ext, 'index': len(images)})
        except Exception as e: st.warning(f"Nie udało się wyekstraktować obrazów z DOCX: {e}")
        return images
    def render_page_as_image(self, page_index: int) -> Optional[bytes]:
        if self.file_type != 'pdf': return None
        try:
            page = self._document.load_page(page_index); pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            return pix.tobytes("png")
        except Exception as e:
            st.error(f"Błąd podczas renderowania strony {page_index + 1}: {e}")
            return None

# ===== POZOSTAŁY KOD APLIKACJI (BEZ ZMIAN) =====
class AIProcessor:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.client = AsyncOpenAI(api_key=api_key); self.model = model
    def get_system_prompt(self) -> str: return """Jesteś precyzyjnym asystentem redakcyjnym...""" # Skrócone dla zwięzłości
    def get_meta_tags_prompt(self) -> str: return """Jesteś ekspertem SEO...""" # Skrócone
    async def process_text(self, text: str, system_prompt: str, max_tokens: int = 4096) -> Dict:
        last_error = None; content = ""
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}], max_tokens=max_tokens, temperature=0.1, response_format={"type": "json_object"})
                content = response.choices[0].message.content
                if not content: raise ValueError("API zwróciło pustą odpowiedź.")
                return json.loads(content)
            except Exception as e:
                last_error = e; await asyncio.sleep(1)
        return {"error": f"Błąd po {MAX_RETRIES} próbach.", "last_known_error": str(last_error), "raw_response": content}
    async def process_page(self, page_content: PageContent) -> Dict:
        page_data = {"page_number": page_content.page_number, "extraction_method": page_content.extraction_method, "ocr_confidence": page_content.ocr_confidence}
        if len(page_content.text.split()) < 20: page_data["type"] = "pominięta"; page_data["formatted_content"] = "<i>Strona zawiera zbyt mało tekstu.</i>"; return page_data
        result = await self.process_text(page_content.text, self.get_system_prompt(), max_tokens=4096)
        if "error" in result: page_data["type"] = "błąd"; page_data["formatted_content"] = f"<div class=\"error-box\"><strong>{result['error']}</strong><br><i>...</i></div>"
        else:
            page_data["type"] = result.get("type", "nieznany").lower(); formatted_text = result.get("formatted_text", "")
            if page_data["type"] == "artykuł": page_data["formatted_content"] = markdown_to_html(formatted_text); page_data["raw_markdown"] = formatted_text
            else: page_data["formatted_content"] = f"<i>Zidentyfikowano jako: <strong>{page_data['type'].upper()}</strong>.</i>"
        return page_data
    async def process_article_group(self, pages_content: List[PageContent]) -> Dict:
        combined_text = "\n\n".join([f"--- STRONA {p.page_number} ---\n{p.text.strip()}" for p in pages_content])
        result = await self.process_text(combined_text, self.get_system_prompt(), max_tokens=8192)
        article_data = {"page_numbers": [p.page_number for p in pages_content]}
        if "error" in result: article_data["type"] = "błąd"; article_data["formatted_content"] = f"<div class='error-box'>...</div>"
        else:
            article_data["type"] = result.get("type", "nieznany").lower(); formatted_text = result.get("formatted_text", "")
            if article_data["type"] == "artykuł": article_data["formatted_content"] = markdown_to_html(formatted_text); article_data["raw_markdown"] = formatted_text
            else: article_data["formatted_content"] = f"<i>Zidentyfikowano jako: <strong>{article_data['type'].upper()}</strong>.</i>"
        return article_data
    def get_optimized_article_prompt(self) -> str: return """Jesteś ekspertem content marketingu i SEO...""" # Skrócone
    async def generate_meta_tags(self, article_text: str) -> Dict: return await self.process_text(article_text[:4000], self.get_meta_tags_prompt(), max_tokens=200)
    async def generate_optimized_article(self, original_markdown: str) -> Dict:
        context = f"Oto zredagowany artykuł do zoptymalizowania:\n\n---\n{original_markdown}\n---\n\nPrzekształć ten artykuł zgodnie z wytycznymi."
        return await self.process_text(context, self.get_optimized_article_prompt(), max_tokens=4096)

def markdown_to_html(text: str) -> str:
    text = re.sub(r'\n---\n', '\n<hr>\n', text)
    text = re.sub(r'^\s*# (.*?)\s*$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*## (.*?)\s*$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*### (.*?)\s*$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    paragraphs = text.split('\n\n'); html_content = []
    for para in paragraphs:
        if para.strip():
            if para.strip().startswith(('<h', '<hr')): html_content.append(para)
            else: html_content.append(f"<p>{para.strip().replace(chr(10), '<br>')}</p>")
    return ''.join(html_content)

# Tutaj reszta funkcji pomocniczych i UI, które zostały bez zmian
# ... (cała reszta kodu, która nie wymagała edycji) ...
def markdown_to_clean_html(markdown_text: str, page_number: int = None) -> str:
    html = markdown_text; html = html.replace('\n---\n', '\n<hr>\n'); html = html.replace('\n--- \n', '\n<hr>\n')
    html = re.sub(r'^\s*# (.*?)\s*$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^\s*## (.*?)\s*$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^\s*### (.*?)\s*$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^\s*#### (.*?)\s*$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    paragraphs = html.split('\n\n'); formatted_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        if para.startswith(('<h1', '<h2', '<h3', '<h4', '<hr', '<p')): formatted_paragraphs.append(para)
        else: para_with_breaks = para.replace('\n', '<br>\n'); formatted_paragraphs.append(f'<p>{para_with_breaks}</p>')
    return '\n'.join(formatted_paragraphs)
def generate_full_html_document(content: str, title: str = "Artykuł", meta_title: str = None, meta_description: str = None) -> str:
    meta_tags = ""
    if meta_title: meta_tags += f'    <meta name="title" content="{meta_title}">\n'
    if meta_description: meta_tags += f'    <meta name="description" content="{meta_description}">\n'
    return f"""<!DOCTYPE html>\n<html lang="pl">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>{title}</title>\n{meta_tags}</head>\n<body>\n{content}\n</body>\n</html>"""
def get_article_html_from_page(page_index: int) -> Optional[Dict]:
    page_result = st.session_state.extracted_pages[page_index]
    if not page_result or page_result.get('type') != 'artykuł' or 'raw_markdown' not in page_result: return None
    group_pages = page_result.get('group_pages', [])
    if group_pages and len(group_pages) > 1:
        first_page_index = group_pages[0] - 1; first_page_result = st.session_state.extracted_pages[first_page_index]
        markdown_content = first_page_result.get('raw_markdown', ''); title = f"Artykuł ze stron {group_pages[0]}-{group_pages[-1]}"; pages = group_pages
    else:
        markdown_content = page_result.get('raw_markdown', ''); title = f"Artykuł ze strony {page_index + 1}"; pages = [page_index + 1]
    html_content = markdown_to_clean_html(markdown_content); meta_title = None; meta_description = None
    if page_index in st.session_state.meta_tags:
        tags = st.session_state.meta_tags[page_index]
        if 'error' not in tags: meta_title = tags.get('meta_title'); meta_description = tags.get('meta_description')
    html_document = generate_full_html_document(html_content, title=title, meta_title=meta_title, meta_description=meta_description)
    return {'html_content': html_content, 'html_document': html_document, 'title': title, 'pages': pages, 'meta_title': meta_title, 'meta_description': meta_description}
def sanitize_filename(name: str) -> str:
    if not name: return "unnamed_project"
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", str(name))
    return re.sub(r'_{2,}', "_", sanitized).strip("_") or "unnamed_project"
def create_zip_archive(data: List[Dict]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in data: zf.writestr(item['name'], item['content'])
    return zip_buffer.getvalue()
def parse_page_groups(input_text: str, total_pages: int) -> List[List[int]]:
    if not input_text: raise ValueError("Nie podano zakresów stron.")
    groups = []; used_pages = set()
    for line in re.split(r'[;\n]+', input_text):
        line = line.strip()
        if not line: continue
        pages = []
        for part in re.split(r'[;,]+', line):
            part = part.strip()
            if not part: continue
            if '-' in part:
                start_str, end_str = part.split('-', 1)
                if not start_str.isdigit() or not end_str.isdigit(): raise ValueError(f"Niepoprawny zakres stron: '{part}'.")
                start, end = int(start_str), int(end_str)
                if start > end: raise ValueError(f"Zakres stron musi być rosnący: '{part}'.")
                if start < 1 or end > total_pages: raise ValueError(f"Zakres '{part}' wykracza poza liczbę stron dokumentu.")
                pages.extend(range(start, end + 1))
            else:
                if not part.isdigit(): raise ValueError(f"Niepoprawny numer strony: '{part}'.")
                page = int(part)
                if page < 1 or page > total_pages: raise ValueError(f"Strona '{page}' wykracza poza dokument.")
                pages.append(page)
        if not pages: continue
        pages = sorted(dict.fromkeys(pages))
        if any(p in used_pages for p in pages): raise ValueError(f"Strony {pages} zostały już przypisane do innego artykułu.")
        used_pages.update(pages); groups.append(pages)
    if not groups: raise ValueError("Nie znaleziono żadnych poprawnych zakresów stron.")
    return groups
def ensure_projects_dir() -> bool:
    try: PROJECTS_DIR.mkdir(exist_ok=True); return True
    except Exception as e: st.error(f"Nie można utworzyć katalogu projektów: {e}"); return False
def get_existing_projects() -> List[str]:
    if not ensure_projects_dir(): return []
    return [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]
def save_project():
    if not st.session_state.project_name or not ensure_projects_dir(): st.error("Nie można zapisać projektu: brak nazwy projektu."); return
    project_path = PROJECTS_DIR / st.session_state.project_name; project_path.mkdir(exist_ok=True)
    state_to_save = {k: v for k, v in st.session_state.items() if k not in ['document', 'project_loaded_and_waiting_for_file'] and not k.startswith("paddleocr_engine_")}
    state_to_save['extracted_pages'] = [p for p in st.session_state.extracted_pages if p is not None]
    try:
        with open(project_path / "project_state.json", "w", encoding="utf-8") as f: json.dump(state_to_save, f, indent=2, ensure_ascii=False)
        st.toast(f"✅ Projekt '{st.session_state.project_name}' został zapisany!", icon="💾")
    except Exception as e: st.error(f"Błąd podczas zapisywania projektu: {e}")
def load_project(project_name: str):
    project_file = PROJECTS_DIR / project_name / "project_state.json"
    if not project_file.exists(): st.error(f"Plik projektu '{project_name}' nie istnieje."); return
    try:
        with open(project_file, "r", encoding="utf-8") as f: state_to_load = json.load(f)
        for key, value in state_to_load.items():
            if key != 'document': st.session_state[key] = value
        total_pages = st.session_state.get('total_pages', 0); st.session_state.extracted_pages = [None] * total_pages
        for page_data in state_to_load.get('extracted_pages', []):
            page_num_one_based = page_data.get('page_number')
            if page_num_one_based and 1 <= page_num_one_based <= total_pages: st.session_state.extracted_pages[page_num_one_based - 1] = page_data
        st.session_state.document = None; st.session_state.project_loaded_and_waiting_for_file = True
        st.success(f"✅ Załadowano projekt '{project_name}'. Wgraj powiązany plik, aby kontynuować.")
    except Exception as e: st.error(f"Błąd podczas ładowania projektu: {e}")
def handle_file_upload(uploaded_file):
    try:
        with st.spinner("Ładowanie pliku..."):
            file_bytes = uploaded_file.read(); document = DocumentHandler(file_bytes, uploaded_file.name)
            if st.session_state.project_loaded_and_waiting_for_file:
                if document.get_page_count() != st.session_state.total_pages: st.error(f"Błąd: Wgrany plik ma {document.get_page_count()} stron, a projekt oczekuje {st.session_state.total_pages}. Wgraj właściwy plik."); return
                st.session_state.document = document; st.session_state.uploaded_filename = uploaded_file.name; st.session_state.file_type = document.file_type; st.session_state.project_loaded_and_waiting_for_file = False; st.success("✅ Plik pomyślnie dopasowany do projektu.")
            else:
                WIDGET_KEYS = {'api_key', 'ocr_mode', 'ocr_language', 'processing_mode', 'start_page', 'end_page', 'article_page_groups_input'}
                for key, value in SESSION_STATE_DEFAULTS.items():
                    if key not in WIDGET_KEYS: st.session_state[key] = value
                st.session_state.document = document; st.session_state.uploaded_filename = uploaded_file.name; st.session_state.file_type = document.file_type; st.session_state.project_name = sanitize_filename(Path(uploaded_file.name).stem); st.session_state.total_pages = document.get_page_count(); st.session_state.extracted_pages = [None] * document.get_page_count(); st.session_state.end_page = document.get_page_count()
                st.success(f"✅ Załadowano plik: {uploaded_file.name} ({document.file_type.upper()})")
    except Exception as e: st.error(f"❌ Błąd ładowania pliku: {e}"); st.session_state.document = None
    st.rerun()
async def process_batch(ai_processor: AIProcessor, start_index: int):
    processing_limit = st.session_state.processing_end_page_index + 1; end_index = min(start_index + BATCH_SIZE, processing_limit); tasks = []
    for i in range(start_index, end_index):
        if st.session_state.document:
            page_content = st.session_state.document.get_page_content(i, force_mode=None); tasks.append(ai_processor.process_page(page_content))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        page_index = start_index + i
        if isinstance(result, Exception): st.session_state.extracted_pages[page_index] = {"page_number": page_index + 1, "type": "błąd", "formatted_content": f"Błąd: {result}"}
        else: st.session_state.extracted_pages[page_index] = result
def start_ai_processing():
    if st.session_state.processing_mode == 'article':
        try:
            groups = parse_page_groups(st.session_state.article_page_groups_input, st.session_state.total_pages)
            for group in groups:
                for page in group: st.session_state.extracted_pages[page - 1] = None
            st.session_state.article_groups = groups; st.session_state.next_article_index = 0; st.session_state.processing_status = 'in_progress'
            if groups: st.session_state.current_page = groups[0][0] - 1
        except ValueError as e: st.error(str(e)); return
    else:
        if st.session_state.processing_mode == 'all': start_idx = 0; end_idx = st.session_state.total_pages - 1
        else: start_idx = st.session_state.start_page - 1; end_idx = st.session_state.end_page - 1
        if start_idx > end_idx: st.error("Strona początkowa nie może być większa niż końcowa."); return
        for i in range(start_idx, end_idx + 1): st.session_state.extracted_pages[i] = None
        st.session_state.processing_status = 'in_progress'; st.session_state.next_batch_start_index = start_idx; st.session_state.processing_end_page_index = end_idx; st.session_state.current_page = start_idx
def run_ai_processing_loop():
    if not st.session_state.api_key: st.error("Klucz API OpenAI nie jest skonfigurowany."); st.session_state.processing_status = 'idle'; return
    ai_processor = AIProcessor(st.session_state.api_key, st.session_state.model)
    if st.session_state.processing_mode == 'article':
        if st.session_state.next_article_index < len(st.session_state.article_groups):
            article_pages = st.session_state.article_groups[st.session_state.next_article_index]; pages_content = []
            for page_num in article_pages:
                if st.session_state.document and 0 <= page_num - 1 < st.session_state.total_pages: pages_content.append(st.session_state.document.get_page_content(page_num - 1, force_mode=None))
            article_result = asyncio.run(ai_processor.process_article_group(pages_content))
            for page in article_pages:
                page_index = page - 1
                if 0 <= page_index < len(st.session_state.extracted_pages):
                    entry = {key: value for key, value in article_result.items() if key != 'page_numbers'}; entry['page_number'] = page; entry['group_pages'] = article_pages; entry['is_group_lead'] = (page == article_pages[0]); st.session_state.extracted_pages[page_index] = entry
            st.session_state.next_article_index += 1
        else: st.session_state.processing_status = 'complete'
    else:
        if st.session_state.next_batch_start_index <= st.session_state.processing_end_page_index: asyncio.run(process_batch(ai_processor, st.session_state.next_batch_start_index)); st.session_state.next_batch_start_index += BATCH_SIZE
        else: st.session_state.processing_status = 'complete'
    st.rerun()
def init_session_state():
    if 'api_key' not in st.session_state or st.session_state.api_key is None: st.session_state.api_key = st.secrets.get("openai", {}).get("api_key")
    for key, value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state: st.session_state[key] = value
# Tutaj cała reszta funkcji UI (render_sidebar, render_page_view, itd.), które są długie, ale nie wymagały zmian.
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Konfiguracja Projektu")
        if PADDLEOCR_AVAILABLE:
            with st.expander("🔍 Silnik Ekstrakcji Tekstu (PaddleOCR)", expanded=True):
                st.info("✨ PaddleOCR jest włączony jako główny silnik!")
                st.radio("Tryb ekstrakcji:", options=['paddleocr', 'auto', 'native'], format_func=lambda x: {'paddleocr': '🔬 PaddleOCR (domyślny - najlepsza jakość)','auto': '🤖 Auto (inteligentny wybór)','native': '📄 PyMuPDF (szybki - tylko natywne PDF)'}[x], key='ocr_mode', help="""• **PaddleOCR**: Zawsze używa OCR - najlepsza jakość, działa na skanach\n• **Auto**: System decyduje - OCR dla skanów, PyMuPDF dla natywnych\n• **PyMuPDF**: Tylko dla nowoczesnych PDF z tekstem (szybkie)""")
                st.selectbox("Język dokumentu:", options=['pl', 'en', 'de', 'fr', 'es', 'it', 'ch_sim', 'ru'], format_func=lambda x: {'pl': '🇵🇱 Polski','en': '🇬🇧 Angielski','de': '🇩🇪 Niemiecki','fr': '🇫🇷 Francuski','es': '🇪🇸 Hiszpański','it': '🇮🇹 Włoski','ch_sim': '🇨🇳 Chiński (uproszczony)','ru': '🇷🇺 Rosyjski'}[x], key='ocr_language')
        else:
            with st.expander("⚠️ PaddleOCR niedostępny", expanded=True): st.warning("PaddleOCR nie jest zainstalowany!"); st.code("pip install paddleocr", language="bash");
        st.divider()
        projects = get_existing_projects(); selected_project = st.selectbox("Wybierz istniejący projekt", ["Nowy projekt"] + projects)
        if st.button("Załaduj projekt", disabled=(selected_project == "Nowy projekt")): load_project(selected_project); st.rerun()
        st.divider()
        supported_formats = ["pdf"]
        if DOCX_AVAILABLE: supported_formats.append("docx")
        if MAMMOTH_AVAILABLE: supported_formats.append("doc")
        file_label = f"Wybierz plik ({', '.join(f.upper() for f in supported_formats)})"; uploaded_file = st.file_uploader(file_label, type=supported_formats)
        if uploaded_file:
            if (st.session_state.project_loaded_and_waiting_for_file or uploaded_file.name != st.session_state.get('uploaded_filename')): handle_file_upload(uploaded_file)
        if st.session_state.document:
            st.divider(); st.subheader("🤖 Opcje Przetwarzania")
            st.radio("Wybierz tryb:", ('all', 'range', 'article'), captions=["Cały dokument", "Zakres stron", "📰 Artykuł wielostronicowy"], key='processing_mode', horizontal=False)
            if st.session_state.processing_mode == 'range':
                c1, c2 = st.columns(2); c1.number_input("Od strony", 1, st.session_state.total_pages, key='start_page'); c2.number_input("Do strony", st.session_state.start_page, st.session_state.total_pages, key='end_page')
            elif st.session_state.processing_mode == 'article':
                st.text_area("Zakresy stron artykułów (jeden na linię)", key='article_page_groups_input', placeholder="2-4\n6-8", height=120)
            st.divider()
            processing_disabled = (st.session_state.processing_status == 'in_progress' or not st.session_state.api_key)
            if st.button("🚀 Rozpocznij Przetwarzanie", use_container_width=True, type="primary", disabled=processing_disabled): start_ai_processing(); st.rerun()
            st.divider(); st.info(f"**Projekt:** {st.session_state.project_name}"); st.metric("Liczba stron", st.session_state.total_pages)
def render_processing_status():
    # ... skrócone dla zwięzłości
    pass
def render_navigation():
    # ... skrócone
    pass
def render_page_view():
    # ... skrócone
    pass
def handle_page_reroll(page_index: int):
    # ... skrócone
    pass
def handle_meta_tag_generation(page_index: int, raw_markdown: str):
    # ... skrócone
    pass
def handle_article_optimization(page_index: int, raw_markdown: str):
    # ... skrócone
    pass

def main():
    st.set_page_config(layout="wide", page_title="Redaktor AI", page_icon="🚀")
    st.markdown("""<style>.page-text-wrapper { ... } .error-box { ... }</style>""", unsafe_allow_html=True) # Skrócone
    st.title("🚀 Redaktor AI - Procesor Dokumentów (PaddleOCR)")
    init_session_state()
    if not PADDLEOCR_AVAILABLE: st.error("⚠️ **Krytyczny błąd: PaddleOCR nie jest zainstalowany!**")
    if not st.session_state.api_key: st.error("❌ Brak klucza API OpenAI!"); st.stop()
    render_sidebar()
    if not st.session_state.document:
        st.info("👋 Witaj! Wgraj plik lub załaduj projekt.")
        return
    render_processing_status()
    if st.session_state.processing_status == 'in_progress': run_ai_processing_loop()
    else: render_navigation(); render_page_view()

if __name__ == "__main__":
    main()
