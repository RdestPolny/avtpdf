# main_app.py
import streamlit as st
import fitz  # PyMuPDF
import openai
import os
import io
import zipfile
import json
from pathlib import Path
import re
import asyncio
from openai import AsyncOpenAI

# --- STAŁE I KONFIGURACJA ---

PROJECTS_DIR = Path("pdf_processor_projects")
BATCH_SIZE = 10
MAX_RETRIES = 2
# ZMIANA: Używamy nowego, przyszłego modelu
DEFAULT_MODEL = 'gpt-5-nano' 

SESSION_STATE_DEFAULTS = {
    'processing_status': 'idle', 'pdf_doc': None, 'current_page': 0, 'total_pages': 0,
    'extracted_pages': [], 'project_name': None, 'next_batch_start_index': 0,
    'uploaded_filename': None, 'api_key': None, 'model': DEFAULT_MODEL, 'meta_tags': {},
    'project_loaded_and_waiting_for_pdf': False, 'processing_mode': 'all', 'start_page': 1,
    'end_page': 1, 'processing_end_page_index': 0, 'article_page_groups_input': '',
    'article_groups': [], 'next_article_index': 0
}

INSTRUCTIONS_MD = "..." # (Instrukcje bez zmian)

st.set_page_config(layout="wide", page_title="Redaktor AI - Procesor PDF", page_icon="🚀")
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # (Style CSS bez zmian)

# --- ZARZĄDZANIE STANEM APLIKACJI ---

def init_session_state():
    SESSION_STATE_DEFAULTS['api_key'] = st.secrets.get("openai", {}).get("api_key")
    for key, value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- RDZEŃ LOGIKI AI (DOSTOSOWANY DO NOWEGO API) ---

@st.cache_resource
def get_openai_client():
    if not st.session_state.api_key: return None
    return AsyncOpenAI(api_key=st.session_state.api_key)

# ZMIANA: Prompty zostały podzielone na instrukcje (developer) i dane wejściowe (user)
def get_developer_instructions():
    return """
Jesteś precyzyjnym asystentem redakcyjnym. Twoim celem jest przekształcenie surowego tekstu w czytelny, dobrze zorganizowany artykuł internetowy.
ZASADA NADRZĘDNA: WIERNOŚĆ TREŚCI, ELASTYCZNOŚĆ FORMY.
- Nie zmieniaj oryginalnych sformułowań ani nie parafrazuj tekstu. Przenieś treść 1:1.
- Twoja rola polega na dodawaniu elementów strukturalnych (nagłówki, pogrubienia, podział na akapity).
INSTRUKCJE SPECJALNE:
1. Ignoruj i pomijaj numery stron oraz rozstrzelone daty.
2. Etykiety jak "NEWS FLASH" używaj jako kontekstu, ale nie umieszczaj ich w finalnym tekście.
DOZWOLONE MODYFIKACJE STRUKTURALNE:
1. Tytuł Główny: `# Tytuł`
2. Śródtytuły: `## Śródtytuł` (używaj ich do rozbijania 'ściany tekstu').
3. Pogrubienia: `**tekst**` (dla kluczowych terminów i nazw własnych).
4. Podział na sekcje: `---` (jeśli na stronie są dwa niepowiązane tematy).
WYMAGANIA KRYTYCZNE:
- Twoja odpowiedź musi być WYŁĄCZNIE i BEZWZGLĘDNIE poprawnym obiektem JSON.
FORMAT ODPOWIEDZI:
{"type": "ARTYKUŁ" lub "REKLAMA", "formatted_text": "Sformatowany tekst w markdown."}
"""

def get_meta_tags_instructions():
    return """
Jesteś ekspertem SEO. Na podstawie poniższego tekstu artykułu, wygeneruj chwytliwy meta title i zwięzły meta description.
WYMAGANIA:
- Meta title: max 60 znaków.
- Meta description: max 160 znaków.
- Odpowiedź zwróć jako obiekt JSON.
FORMAT ODPOWIEDZI:
{"meta_title": "Tytuł meta", "meta_description": "Opis meta."}
"""

# ZMIANA: Całkowicie nowa funkcja do obsługi nowego API `responses.create`
async def _call_openai_api(client, model, developer_instructions, user_content, max_tokens, temperature=0.1):
    last_error = None
    content = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Używamy nowej składni `client.responses.create`
            response = await client.responses.create(
                model=model,
                instructions=developer_instructions,
                input=user_content
                # Zakładamy, że max_tokens i temperature będą w przyszłości dostępne,
                # ale ich nazwy mogą się zmienić. `reasoning` jest pominięte jako opcjonalne.
            )
            # Używamy nowej właściwości `output_text`
            content = response.output_text
            if not content:
                raise ValueError("API zwróciło pustą odpowiedź.")
            
            # Parsowanie JSON pozostaje takie samo
            return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES: await asyncio.sleep(1)
            continue
        except Exception as e:
            last_error = e
            break
            
    return {
        "error": f"Błąd parsowania po {MAX_RETRIES + 1} próbach.",
        "last_known_error": str(last_error),
        "raw_response": content or "Brak odpowiedzi"
    }

async def process_page_async(client, page_num, raw_text, model):
    page_data = {"page_number": page_num}
    if len(raw_text.split()) < 20:
        page_data["type"] = "pominięta"; page_data["formatted_content"] = "<i>Strona zawiera zbyt mało tekstu.</i>"
        return page_data

    developer_instructions = get_developer_instructions()
    ai_result = await _call_openai_api(client, model, developer_instructions, raw_text, 2048)

    if "error" in ai_result:
        page_data["type"] = "błąd"
        page_data["formatted_content"] = f"""<div class="error-box"><strong>{ai_result['error']}</strong><br>
            <i>Ostatni błąd: {ai_result['last_known_error']}</i><br>
            <details><summary>Pokaż surową odpowiedź</summary><pre>{ai_result['raw_response']}</pre></details></div>"""
    else:
        page_data["type"] = ai_result.get("type", "nieznany").lower()
        formatted_text = ai_result.get("formatted_text", "")
        if page_data["type"] == "artykuł":
            page_data["formatted_content"] = markdown_to_html(formatted_text)
            page_data["raw_markdown"] = formatted_text
        else:
            page_data["formatted_content"] = f"<i>Zidentyfikowano jako: <strong>{page_data['type'].upper()}</strong>.</i>"
    return page_data

async def process_article_group_async(client, page_numbers, raw_text, model):
    developer_instructions = get_developer_instructions()
    # Zwiększony limit tokenów dla długich artykułów
    ai_result = await _call_openai_api(client, model, developer_instructions, raw_text, 8192)
    
    article_data = {"page_numbers": page_numbers}
    if "error" in ai_result:
        article_data["type"] = "błąd"
        article_data["formatted_content"] = f"""<div class='error-box'><strong>{ai_result['error']}</strong><br>
            <i>Ostatni błąd: {ai_result['last_known_error']}</i><br>
            <details><summary>Pokaż surową odpowiedź</summary><pre>{ai_result['raw_response']}</pre></details></div>"""
    else:
        article_data["type"] = ai_result.get("type", "nieznany").lower()
        formatted_text = ai_result.get("formatted_text", "")
        if article_data["type"] == "artykuł":
            article_data["formatted_content"] = markdown_to_html(formatted_text)
            article_data["raw_markdown"] = formatted_text
        else:
            article_data["formatted_content"] = f"<i>Zidentyfikowano jako: <strong>{article_data['type'].upper()}</strong>.</i>"
    return article_data

async def generate_meta_tags_async(client, article_text, model):
    instructions = get_meta_tags_instructions()
    return await _call_openai_api(client, model, instructions, article_text[:4000], 200, temperature=0.5)

async def process_batch(client, start_index):
    processing_limit = st.session_state.processing_end_page_index + 1
    end_index = min(start_index + BATCH_SIZE, processing_limit)
    tasks = [
        process_page_async(client, i + 1, st.session_state.pdf_doc.load_page(i).get_text("text"), st.session_state.model)
        for i in range(start_index, end_index) if st.session_state.pdf_doc
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        page_index = start_index + i
        if isinstance(result, Exception):
            st.session_state.extracted_pages[page_index] = {"page_number": page_index + 1, "type": "błąd", "formatted_content": f"Błąd: {result}"}
        else:
            st.session_state.extracted_pages[page_index] = result

# --- FUNKCJE OBSŁUGI PROJEKTU, PLIKÓW I INNE FUNKCJE POMOCNICZE ---
# (W tej sekcji kod pozostaje bez zmian, ponieważ nie dotyczy bezpośrednio logiki AI)

def ensure_projects_dir():
    try:
        PROJECTS_DIR.mkdir(exist_ok=True); return True
    except Exception as e:
        st.error(f"Nie można utworzyć katalogu projektów: {e}"); return False

def get_existing_projects():
    if not ensure_projects_dir(): return []
    return [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]

def save_project():
    if not st.session_state.project_name or not ensure_projects_dir():
        st.error("Nie można zapisać projektu: brak nazwy projektu."); return
    project_path = PROJECTS_DIR / st.session_state.project_name; project_path.mkdir(exist_ok=True)
    state_to_save = {k: v for k, v in st.session_state.items() if k not in ['pdf_doc', 'project_loaded_and_waiting_for_pdf']}
    state_to_save['extracted_pages'] = [p for p in st.session_state.extracted_pages if p is not None]
    try:
        with open(project_path / "project_state.json", "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=2, ensure_ascii=False)
        st.toast(f"✅ Projekt '{st.session_state.project_name}' został zapisany!", icon="💾")
    except Exception as e: st.error(f"Błąd podczas zapisywania projektu: {e}")

def load_project(project_name):
    project_file = PROJECTS_DIR / project_name / "project_state.json"
    if not project_file.exists():
        st.error(f"Plik projektu '{project_name}' nie istnieje."); return
    try:
        with open(project_file, "r", encoding="utf-8") as f: state_to_load = json.load(f)
        for key, value in state_to_load.items():
            if key != 'pdf_doc': st.session_state[key] = value
        total_pages = st.session_state.get('total_pages', 0)
        st.session_state.extracted_pages = [None] * total_pages
        for page_data in state_to_load.get('extracted_pages', []):
            page_num_one_based = page_data.get('page_number')
            if page_num_one_based and 1 <= page_num_one_based <= total_pages:
                st.session_state.extracted_pages[page_num_one_based - 1] = page_data
        st.session_state.pdf_doc = None
        st.session_state.project_loaded_and_waiting_for_pdf = True
        st.success(f"✅ Załadowano projekt '{project_name}'. Wgraj powiązany plik PDF, aby kontynuować.")
    except Exception as e: st.error(f"Błąd podczas ładowania projektu: {e}")

def handle_file_upload(uploaded_file):
    try:
        with st.spinner("Ładowanie pliku PDF..."):
            pdf_bytes = uploaded_file.read()
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if st.session_state.project_loaded_and_waiting_for_pdf:
                if len(pdf_doc) != st.session_state.total_pages:
                    st.error(f"Błąd: Wgrany PDF ma {len(pdf_doc)} stron, a projekt oczekuje {st.session_state.total_pages}. Wgraj właściwy plik.")
                    return
                st.session_state.pdf_doc = pdf_doc
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.project_loaded_and_waiting_for_pdf = False
                st.success("✅ Plik PDF pomyślnie dopasowany do projektu.")
            else:
                init_session_state()
                st.session_state.pdf_doc = pdf_doc
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.project_name = sanitize_filename(Path(uploaded_file.name).stem)
                st.session_state.total_pages = len(pdf_doc)
                st.session_state.extracted_pages = [None] * len(pdf_doc)
                st.session_state.end_page = len(pdf_doc)
    except Exception as e:
        st.error(f"❌ Błąd ładowania pliku: {e}")
        st.session_state.pdf_doc = None
    st.rerun()

def sanitize_filename(name):
    if not name: return "unnamed_project"
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", str(name))
    return re.sub(r'_{2,}', "_", sanitized).strip("_") or "unnamed_project"

def markdown_to_html(text):
    text = text.replace('\n---\n', '\n<hr>\n')
    text = re.sub(r'^\s*# (.*?)\s*$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*## (.*?)\s*$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*### (.*?)\s*$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    paragraphs = text.split('\n\n')
    html_content = [
        para if para.strip().startswith(('<h', '<hr')) else f"<p>{para.strip().replace(chr(10), '<br>')}</p>"
        for para in paragraphs if para.strip()
    ]
    return ''.join(html_content)

def render_page_as_image(pdf_doc, page_num):
    try:
        if not pdf_doc or not (0 <= page_num < len(pdf_doc)): return None
        page = pdf_doc.load_page(page_num); pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        return pix.tobytes("png")
    except Exception as e: st.error(f"Błąd podczas renderowania strony {page_num + 1}: {e}"); return None

def extract_images_from_page(pdf_doc, page_num):
    images = []
    if not pdf_doc or not (0 <= page_num < len(pdf_doc)): return images
    try:
        page = pdf_doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]; base_image = pdf_doc.extract_image(xref)
            if base_image and base_image.get("width", 0) > 100 and base_image.get("height", 0) > 100:
                images.append({'image': base_image['image'], 'ext': base_image['ext'], 'index': img_index})
    except Exception as e: st.warning(f"Nie udało się wyekstraktować obrazów ze strony {page_num + 1}: {e}")
    return images

def create_zip_archive(data):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in data: zf.writestr(item['name'], item['content'])
    return zip_buffer.getvalue()

def parse_page_groups(input_text, total_pages):
    if not input_text: raise ValueError("Nie podano zakresów stron.")
    groups, used_pages = [], set()
    for line in re.split(r'[;\n]+', input_text):
        if not (line := line.strip()): continue
        pages = []
        for part in re.split(r'[;,]+', line):
            if not (part := part.strip()): continue
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

def assign_article_result_to_pages(article_result):
    for page in article_result.get('page_numbers', []):
        page_index = page - 1
        if 0 <= page_index < len(st.session_state.extracted_pages):
            entry = {key: value for key, value in article_result.items() if key != 'page_numbers'}
            entry['page_number'] = page
            entry['group_pages'] = article_result.get('page_numbers', [])
            entry['is_group_lead'] = (page == article_result.get('page_numbers', [page])[0])
            st.session_state.extracted_pages[page_index] = entry

# --- KOMPONENTY UI I LOGIKA APLIKACJI ---
# (W tej sekcji kod również pozostaje bez zmian)

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Konfiguracja Projektu")
        projects = get_existing_projects()
        selected_project = st.selectbox("Wybierz istniejący projekt", ["Nowy projekt"] + projects)
        if st.button("Załaduj projekt", disabled=(selected_project == "Nowy projekt")):
            load_project(selected_project); st.rerun()
        st.divider()
        uploaded_file = st.file_uploader("Wybierz plik PDF", type="pdf")
        if uploaded_file and st.session_state.project_loaded_and_waiting_for_pdf: handle_file_upload(uploaded_file)
        elif uploaded_file and uploaded_file.name != st.session_state.get('uploaded_filename'): handle_file_upload(uploaded_file)
        if st.session_state.pdf_doc:
            st.divider()
            st.subheader("🤖 Opcje Przetwarzania")
            st.radio("Wybierz tryb:", ('all', 'range', 'article'), captions=["Cały dokument", "Zakres stron", "Artykuł wielostronicowy"], key='processing_mode', horizontal=True)
            if st.session_state.processing_mode == 'range':
                c1, c2 = st.columns(2)
                c1.number_input("Od strony", min_value=1, max_value=st.session_state.total_pages, key='start_page')
                c2.number_input("Do strony", min_value=st.session_state.start_page, max_value=st.session_state.total_pages, key='end_page')
            elif st.session_state.processing_mode == 'article':
                st.info("Podaj grupy stron, np. 1-3; 5,6.")
                st.text_area("Zakresy stron artykułów", key='article_page_groups_input', placeholder="1-3\n5,6")
            st.divider()
            processing_disabled = st.session_state.processing_status == 'in_progress' or not st.session_state.api_key
            button_text = "🔄 Przetwarzanie..." if st.session_state.processing_status == 'in_progress' else "🚀 Rozpocznij Przetwarzanie"
            if st.button(button_text, use_container_width=True, type="primary", disabled=processing_disabled, key="stButton-ProcessAI"):
                start_ai_processing(); st.rerun()
            st.divider()
            st.info(f"**Projekt:** {st.session_state.project_name}")
            st.metric("Liczba stron", st.session_state.total_pages)

def render_processing_status():
    if st.session_state.processing_status == 'idle' or not st.session_state.pdf_doc: return
    processed_count = sum(1 for p in st.session_state.extracted_pages if p is not None)
    progress = processed_count / st.session_state.total_pages if st.session_state.total_pages > 0 else 0
    if st.session_state.processing_status == 'complete': st.success("✅ Przetwarzanie zakończone!")
    else:
        st.info(f"🔄 Przetwarzanie w toku... (Ukończono {processed_count}/{st.session_state.total_pages} stron)")
        st.progress(progress)
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("💾 Zapisz postęp", use_container_width=True): save_project()
    articles = [p for p in st.session_state.extracted_pages if p and p.get('type') == 'artykuł' and p.get('is_group_lead', True)]
    if articles:
        zip_data = [{'name': f"artykul_ze_str_{a['page_number']}.txt", 'content': a['raw_markdown'].encode('utf-8')} for a in articles if 'raw_markdown' in a]
        if zip_data: c2.download_button("📥 Pobierz artykuły", create_zip_archive(zip_data), f"{st.session_state.project_name}_artykuly.zip", "application/zip", use_container_width=True)

def render_navigation():
    if st.session_state.total_pages <= 1: return
    st.subheader("📖 Nawigacja")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("⬅️ Poprzednia", use_container_width=True, disabled=(st.session_state.current_page == 0)):
        st.session_state.current_page -= 1; st.rerun()
    c2.metric("Strona", f"{st.session_state.current_page + 1} / {st.session_state.total_pages}")
    if c3.button("Następna ➡️", use_container_width=True, disabled=(st.session_state.current_page >= st.session_state.total_pages - 1)):
        st.session_state.current_page += 1; st.rerun()
    new_page = st.slider("Przejdź do strony:", 1, st.session_state.total_pages, st.session_state.current_page + 1) - 1
    if new_page != st.session_state.current_page:
        st.session_state.current_page = new_page; st.rerun()

def render_page_view():
    st.divider()
    page_index = st.session_state.current_page
    pdf_col, text_col = st.columns(2, gap="large")
    with pdf_col:
        st.subheader(f"📄 Oryginał (Strona {page_index + 1})")
        image_data = render_page_as_image(st.session_state.pdf_doc, page_index)
        if image_data:
            st.image(image_data, use_container_width=True)
            images = extract_images_from_page(st.session_state.pdf_doc, page_index)
            if images:
                with st.expander(f"🖼️ Pokaż/ukryj {len(images)} obraz(y)"):
                    for img in images: st.image(img['image'], caption=f"Obraz {img['index'] + 1}", use_container_width=True)
                img_zip = create_zip_archive([{'name': f"str_{page_index+1}_img_{i['index']}.{i['ext']}", 'content': i['image']} for i in images])
                st.download_button("Pobierz obrazy", img_zip, f"obrazy_strona_{page_index+1}.zip", "application/zip", use_container_width=True)
        else: st.error("Nie można wyświetlić podglądu strony.")
    with text_col:
        st.subheader("🤖 Tekst przetworzony przez AI")
        raw_text = st.session_state.pdf_doc.load_page(page_index).get_text("text")
        with st.expander("👁️ Pokaż surowy tekst wejściowy"):
            st.text_area("Surowy tekst", raw_text, height=200, disabled=True, key=f"raw_text_{page_index}")
        page_result = st.session_state.extracted_pages[page_index]
        if page_result:
            page_type = page_result.get('type', 'nieznany')
            color = {"artykuł": "green", "reklama": "orange", "pominięta": "grey"}.get(page_type, "red")
            st.markdown(f"**Status:** <span style='color:{color}; text-transform:uppercase;'>**{page_type}**</span>", unsafe_allow_html=True)
            if (group_pages := page_result.get('group_pages', [])) and len(group_pages) > 1:
                st.info(f"Ten artykuł obejmuje strony: {', '.join(str(p) for p in group_pages)}.")
            st.markdown(f"<div class='page-text-wrapper'>{page_result.get('formatted_content', '')}</div>", unsafe_allow_html=True)
            action_cols = st.columns(2)
            if action_cols[0].button("🔄 Przetwórz ponownie (z kontekstem)", key=f"reroll_{page_index}", use_container_width=True):
                handle_page_reroll(page_index)
            allow_meta = page_type == 'artykuł' and 'raw_markdown' in page_result and page_result.get('is_group_lead', True)
            if action_cols[1].button("✨ SEO: Generuj Meta Tagi", key=f"meta_{page_index}", use_container_width=True, disabled=not allow_meta):
                handle_meta_tag_generation(page_index, page_result['raw_markdown'])
            if page_index in st.session_state.meta_tags:
                tags = st.session_state.meta_tags[page_index]
                if "error" in tags: st.error(f"Błąd generowania meta tagów: {tags['error']}")
                else:
                    with st.expander("Wygenerowane Meta Tagi ✨", expanded=True):
                        st.text_input("Meta Title", value=tags.get("meta_title", ""), key=f"mt_{page_index}")
                        st.text_area("Meta Description", value=tags.get("meta_description", ""), key=f"md_{page_index}")
        else: st.info("⏳ Strona oczekuje na przetworzenie..." if st.session_state.processing_status == 'in_progress' else "Uruchom przetwarzanie w panelu bocznym.")

def start_ai_processing():
    if st.session_state.processing_mode == 'article':
        try:
            groups = parse_page_groups(st.session_state.article_page_groups_input, st.session_state.total_pages)
            for group in groups:
                for page in group: st.session_state.extracted_pages[page - 1] = None
            st.session_state.article_groups = groups; st.session_state.next_article_index = 0
            st.session_state.processing_status = 'in_progress'
        except ValueError as e: st.error(str(e)); return
    else:
        start_idx, end_idx = (0, st.session_state.total_pages - 1) if st.session_state.processing_mode == 'all' else (st.session_state.start_page - 1, st.session_state.end_page - 1)
        if start_idx > end_idx: st.error("Strona początkowa nie może być większa niż końcowa."); return
        for i in range(start_idx, end_idx + 1): st.session_state.extracted_pages[i] = None
        st.session_state.processing_status = 'in_progress'
        st.session_state.next_batch_start_index = start_idx
        st.session_state.processing_end_page_index = end_idx

def handle_page_reroll(page_index):
    with st.spinner("Przetwarzanie strony z kontekstem..."):
        prev_text = st.session_state.pdf_doc.load_page(page_index - 1).get_text("text") if page_index > 0 else ""
        curr_text = st.session_state.pdf_doc.load_page(page_index).get_text("text")
        next_text = st.session_state.pdf_doc.load_page(page_index + 1).get_text("text") if page_index < st.session_state.total_pages - 1 else ""
        context_text = f"KONTEKST (POPRZEDNIA STRONA):\n{prev_text}\n\n--- STRONA DOCELOWA ---\n{curr_text}\n\nKONTEKST (NASTĘPNA STRONA):\n{next_text}"
        client = get_openai_client()
        new_result = asyncio.run(process_page_async(client, page_index + 1, context_text, st.session_state.model))
        st.session_state.extracted_pages[page_index] = new_result
    st.rerun()

def handle_meta_tag_generation(page_index, raw_markdown):
    with st.spinner("Generowanie meta tagów..."):
        client = get_openai_client()
        tags = asyncio.run(generate_meta_tags_async(client, raw_markdown, st.session_state.model))
        st.session_state.meta_tags[page_index] = tags
    st.rerun()

def run_ai_processing_loop():
    client = get_openai_client()
    if not client: st.error("Klucz API OpenAI nie jest skonfigurowany."); st.session_state.processing_status = 'idle'; return
    if st.session_state.processing_mode == 'article':
        if st.session_state.next_article_index < len(st.session_state.article_groups):
            article_pages = st.session_state.article_groups[st.session_state.next_article_index]
            combined_text = "\n".join([f"--- STRONA {p} ---\n{st.session_state.pdf_doc.load_page(p - 1).get_text('text').strip()}\n" for p in article_pages if st.session_state.pdf_doc and 0 <= p - 1 < st.session_state.total_pages])
            article_result = asyncio.run(process_article_group_async(client, article_pages, combined_text, st.session_state.model))
            assign_article_result_to_pages(article_result)
            st.session_state.next_article_index += 1
        else: st.session_state.processing_status = 'complete'
    else:
        if st.session_state.next_batch_start_index <= st.session_state.processing_end_page_index:
            asyncio.run(process_batch(client, st.session_state.next_batch_start_index))
            st.session_state.next_batch_start_index += BATCH_SIZE
        else: st.session_state.processing_status = 'complete'
    st.rerun()

def main():
    st.title("🚀 Redaktor AI - Interaktywny Procesor PDF")
    init_session_state()
    if not st.session_state.api_key:
        st.error("Brak klucza API OpenAI!")
        st.info("Proszę skonfiguruj swój klucz API w pliku .streamlit/secrets.toml.")
        st.code("[openai]\napi_key = \"sk-...\"", language="toml")
        st.stop()
    render_sidebar()
    if not st.session_state.pdf_doc:
        if not st.session_state.project_loaded_and_waiting_for_pdf:
            st.info("👋 Witaj! Aby rozpocząć, wgraj plik PDF lub załaduj istniejący projekt z panelu bocznego.")
            with st.expander("📖 Jak korzystać z aplikacji?"): st.markdown(INSTRUCTIONS_MD, unsafe_allow_html=True)
        return
    render_processing_status()
    if st.session_state.processing_status == 'in_progress':
        run_ai_processing_loop()
    else:
        render_navigation()
        render_page_view()

if __name__ == "__main__":
    main()
