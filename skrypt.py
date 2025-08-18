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

# --- Konfiguracja strony ---
st.set_page_config(
    layout="wide",
    page_title="Redaktor AI - Procesor PDF",
    page_icon="🚀"
)

# --- Style CSS dla nowoczesnego wyglądu ---
st.markdown("""
<style>
    .stApp { 
        background-color: #F0F2F5; 
        font-family: 'Segoe UI', sans-serif; 
    }
    .main > div { 
        background-color: #FFFFFF; 
        padding: 2rem; 
        border-radius: 15px; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1); 
    }
    .stButton>button { 
        border-radius: 8px; 
        border: 1px solid #E0E0E0; 
        color: #333333; 
        background-color: #FFFFFF; 
        transition: all 0.2s ease-in-out; 
        padding: 0.5rem 1rem; 
    }
    .stButton>button:hover { 
        border-color: #007BFF; 
        color: #007BFF; 
        box-shadow: 0 2px 5px 0 rgba(0, 123, 255, 0.2); 
    }
    .stButton[data-testid="stButton-ProcessAI"] button { 
        background-color: #007BFF; 
        color: white; 
        font-weight: bold; 
    }
    .page-text-wrapper { 
        padding: 1.5rem; 
        border-radius: 8px; 
        border: 1px solid #E8E8E8; 
        background-color: #FAFAFA; 
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
    }
    .page-text-wrapper h4 { 
        color: #007BFF; 
        border-bottom: 2px solid #007BFF; 
        padding-bottom: 8px; 
        margin-bottom: 1rem; 
    }
    .stMetric {
        text-align: center;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Zarządzanie stanem sesji ---
def init_session_state():
    defaults = {
        'processing_status': 'idle', 'pdf_doc': None, 'current_page': 0,
        'total_pages': 0, 'extracted_pages': [], 'project_name': None,
        'next_batch_start_index': 0, 'uploaded_filename': None,
        'api_key': os.environ.get("OPENAI_API_KEY", ""), 
        'model': 'gpt-4o-mini',  # ZMIANA: Ustawienie nowego domyślnego modelu
        'meta_tags': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

init_session_state()

# --- Funkcje Pomocnicze ---
PROJECTS_DIR = Path("pdf_processor_projects")
BATCH_SIZE = 10
MAX_RETRIES = 2

def ensure_projects_dir():
    try:
        PROJECTS_DIR.mkdir(exist_ok=True); return True
    except Exception as e:
        st.error(f"Nie można utworzyć katalogu projektów: {e}"); return False

def get_existing_projects():
    if not ensure_projects_dir(): return []
    return [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]

def sanitize_filename(name):
    if not name: return "unnamed_project"
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", str(name))
    return re.sub(r'_{2,}', "_", sanitized).strip("_") or "unnamed_project"

def render_page_as_image(pdf_doc, page_num):
    try:
        if not pdf_doc or not (0 <= page_num < len(pdf_doc)): return None
        page = pdf_doc.load_page(page_num); pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        return pix.tobytes("png")
    except Exception as e:
        st.error(f"Błąd podczas renderowania strony {page_num + 1}: {e}"); return None

def extract_images_from_page(pdf_doc, page_num):
    images = []
    if not pdf_doc or not (0 <= page_num < len(pdf_doc)): return images
    try:
        page = pdf_doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]; base_image = pdf_doc.extract_image(xref)
            if base_image and base_image.get("width", 0) > 100 and base_image.get("height", 0) > 100:
                images.append({'image': base_image['image'], 'ext': base_image['ext'], 'index': img_index})
    except Exception as e:
        st.warning(f"Nie udało się wyekstraktować obrazów ze strony {page_num + 1}: {e}")
    return images

def create_zip_archive(data, file_prefix="item"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in data: zf.writestr(item['name'], item['content'])
    return zip_buffer.getvalue()

def save_project():
    if not st.session_state.project_name or not ensure_projects_dir():
        st.error("Nie można zapisać projektu: brak nazwy projektu."); return
    project_path = PROJECTS_DIR / st.session_state.project_name; project_path.mkdir(exist_ok=True)
    state_to_save = {k: v for k, v in st.session_state.items() if k not in ['pdf_doc']}
    state_to_save['extracted_pages'] = [p for p in st.session_state.extracted_pages if p is not None]
    try:
        with open(project_path / "project_state.json", "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=2, ensure_ascii=False)
        st.toast(f"✅ Projekt '{st.session_state.project_name}' został zapisany!", icon="💾")
    except Exception as e:
        st.error(f"Błąd podczas zapisywania projektu: {e}")

def load_project(project_name):
    project_file = PROJECTS_DIR / project_name / "project_state.json"
    if not project_file.exists(): st.error(f"Plik projektu '{project_name}' nie istnieje."); return
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
        st.success(f"✅ Załadowano projekt '{project_name}'. Wgraj powiązany plik PDF, aby kontynuować.")
        st.session_state.pdf_doc = None
    except Exception as e:
        st.error(f"Błąd podczas ładowania projektu: {e}")

# --- Funkcje Przetwarzania AI ---
def markdown_to_html(text):
    text = text.replace('\n---\n', '\n<hr>\n')
    text = re.sub(r'^\s*# (.*?)\s*$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*## (.*?)\s*$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*### (.*?)\s*$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    paragraphs = text.split('\n\n')
    html_content = []
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        if para.startswith(('<h', '<hr')): html_content.append(para)
        else: html_content.append(f"<p>{para.replace(chr(10), '<br>')}</p>")
    return ''.join(html_content)

async def process_page_async(client, page_num, raw_text, model):
    page_data = {"page_number": page_num}

    if len(raw_text.split()) < 20:
        page_data["type"] = "pominięta"; page_data["formatted_content"] = "<i>Strona zawiera zbyt mało tekstu.</i>"
        return page_data

    prompt = f"""Jesteś precyzyjnym asystentem redakcyjnym. Twoim celem jest przekształcenie surowego tekstu w czytelny, dobrze zorganizowany artykuł internetowy.

ZASADA NADRZĘDNA: WIERNOŚĆ TREŚCI, ELASTYCZNOŚĆ FORMY
- **Nie zmieniaj oryginalnych sformułowań ani nie parafrazuj tekstu.** Twoim zadaniem jest przenieść treść 1:1.
- Twoja rola polega na **dodawaniu elementów strukturalnych** (nagłówki, pogrubienia, podział na akapity) w celu poprawy czytelności.

INSTRUKCJE SPECJALNE: Oczyszczanie i Kontekst
1.  **Ignorowanie Szumu**: Musisz zidentyfikować i całkowicie pominąć w finalnym tekście następujące elementy, jeśli występują na początku lub końcu:
    - Numery stron (np. "6", "12").
    - Rozstrzelone daty (np. "c z e r w i e c  2 0 2 5").
2.  **Wykorzystanie Kategorii**:
    - Na początku tekstu możesz znaleźć etykietę, np. "od redakcji", "NEWS FLASH", "WYWIAD MIESIĄCA".
    - Użyj tej etykiety jako **kontekstu** do zrozumienia intencji tekstu, ale **nie umieszczaj jej w sformatowanym artykule**.

DOZWOLONE MODYFIKACJE STRUKTURALNE:
1.  **Tytuł Główny (`# Tytuł`)**:
    - **ZNAJDŹ go w tekście**. To często krótka linia bez kropki na końcu.
    - **Absolutny zakaz** wymyślania tytułów. Jeśli go nie ma, nie dodawaj go.
2.  **Śródtytuły (`## Śródtytuł`) - Twój kluczowy obowiązek**:
    - Celem jest przekształcenie 'ściany tekstu' w czytelny artykuł.
    - Gdy tekst zawiera kilka następujących po sobie akapitów omawiających różne przykłady, technologie lub firmy, **MUSISZ** je rozdzielić trafnymi, zwięzłymi śródtytułami.
3.  **Pogrubienia (`**tekst**`)**:
    - Stosuj je, by wyróżnić **kluczowe terminy, nazwy własne i frazy ważne dla SEO**.
    - W dłuższych akapitach pogrub **jedno kluczowe zdanie lub wniosek**, aby ułatwić szybkie czytanie.
4.  **Podział na sekcje**:
     - Jeśli tekst na stronie wyraźnie zawiera **dwa lub więcej niepowiązanych ze sobą tematów**, oddziel je linią horyzontalną (`---`).

WYMAGANIA KRYTYCZNE:
- Twoja odpowiedź musi być **WYŁĄCZNIE i BEZWZGLĘDNIE** poprawnym obiektem JSON.

FORMAT ODPOWIEDZI:
{{"type": "ARTYKUŁ" lub "REKLAMA", "formatted_text": "Sformatowany tekst w markdown. Jeśli jest wiele artykułów, oddziel je '---'."}}

TEKST DO PRZETWORZENIA:
---
{raw_text}
---
"""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        content = ""
        try:
            # ZMIANA: Inteligentny przełącznik API w zależności od modelu
            if "gpt-5" in model:
                response = await client.responses.create(
                    model=model, input=prompt,
                    reasoning={"effort": "low"}, text={"verbosity": "low"},
                )
                content = response.output_text
            else: # Domyślne dla GPT-4o, GPT-4, etc.
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=2048
                )
                content = response.choices[0].message.content

            if not content: raise ValueError("API zwróciło pustą odpowiedź.")
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match: raise ValueError("W odpowiedzi AI nie znaleziono obiektu JSON.")
            clean_content = json_match.group(0)
            ai_result = json.loads(clean_content)
            page_data["type"] = ai_result.get("type", "nieznany").lower()
            formatted_text = ai_result.get("formatted_text", "")
            if page_data["type"] == "artykuł":
                page_data["formatted_content"] = markdown_to_html(formatted_text)
                page_data["raw_markdown"] = formatted_text
            else:
                page_data["formatted_content"] = f"<i>Zidentyfikowano jako: <strong>{page_data['type'].upper()}</strong>.</i>"
            return page_data
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES: await asyncio.sleep(1)
            continue
        except Exception as e:
            last_error = e; break
    page_data["type"] = "błąd"
    page_data["formatted_content"] = f"""
    <div class="error-box"><strong>Błąd parsowania po {MAX_RETRIES + 1} próbach.</strong><br><i>Ostatni błąd: {last_error}</i><br>
    <details><summary>Pokaż ostatnią surową odpowiedź</summary><pre>{content or "Brak odpowiedzi"}</pre></details></div>"""
    return page_data

async def generate_meta_tags_async(client, article_text, model):
    prompt = f"""Jesteś ekspertem SEO. Na podstawie poniższego tekstu artykułu, wygeneruj chwytliwy meta title i zwięzły meta description.
WYMAGANIA:
- Meta title: max 60 znaków.
- Meta description: max 160 znaków.
- Odpowiedź zwróć jako obiekt JSON.
FORMAT ODPOWIEDZI:
{{"meta_title": "Tytuł meta", "meta_description": "Opis meta."}}
TEKST ARTYKUŁU:
---
{article_text[:4000]}
---
"""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        content = ""
        try:
            # ZMIANA: Inteligentny przełącznik API w zależności od modelu
            if "gpt-5" in model:
                response = await client.responses.create(
                    model=model, input=prompt,
                    reasoning={"effort": "low"}, text={"verbosity": "low"},
                )
                content = response.output_text
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.5,
                    max_tokens=200
                )
                content = response.choices[0].message.content

            if not content: raise ValueError("API zwróciło pustą odpowiedź.")
            return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES: await asyncio.sleep(1)
            continue
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"Błąd parsowania po {MAX_RETRIES + 1} próbach. Odpowiedź: '{content}'"}

async def process_batch(start_index):
    client = AsyncOpenAI(api_key=st.session_state.api_key)
    end_index = min(start_index + BATCH_SIZE, st.session_state.total_pages)
    tasks = [process_page_async(client, i + 1, st.session_state.pdf_doc.load_page(i).get_text("text"), st.session_state.model) for i in range(start_index, end_index) if st.session_state.pdf_doc]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        page_index = start_index + i
        if isinstance(result, Exception):
            st.session_state.extracted_pages[page_index] = {"page_number": page_index + 1, "type": "błąd", "formatted_content": f"Błąd: {result}"}
        else:
            st.session_state.extracted_pages[page_index] = result

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Konfiguracja Projektu")
        st.subheader("📂 Projekt")
        projects = get_existing_projects()
        selected_project = st.selectbox("Wybierz istniejący projekt", ["Nowy projekt"] + projects)
        if st.button("Załaduj projekt", disabled=(selected_project == "Nowy projekt")):
            load_project(selected_project); st.rerun()
        st.divider()
        st.subheader("📄 Plik PDF")
        uploaded_file = st.file_uploader("Wybierz plik PDF", type="pdf")
        if uploaded_file and uploaded_file.name != st.session_state.uploaded_filename:
            handle_file_upload(uploaded_file)
        st.divider()
        st.subheader("🤖 Konfiguracja AI")
        st.session_state.api_key = st.text_input("Klucz API OpenAI", type="password", value=st.session_state.api_key)
        
        # ZMIANA: Zaktualizowana lista modeli z gpt-4o-mini na początku
        st.session_state.model = st.selectbox(
            "Model AI", 
            ["gpt-4o-mini", "gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5"], 
            index=0
        )
        
        if st.session_state.pdf_doc:
            st.divider()
            processing_disabled = st.session_state.processing_status == 'in_progress' or not st.session_state.api_key
            button_text = "🔄 Przetwarzanie..." if st.session_state.processing_status == 'in_progress' else "🚀 Rozpocznij Przetwarzanie"
            if st.button(button_text, use_container_width=True, type="primary", disabled=processing_disabled, key="stButton-ProcessAI"):
                start_ai_processing(); st.rerun()
        
        if st.session_state.project_name:
            st.divider(); st.info(f"**Projekt:** `{st.session_state.project_name}`"); st.metric("Liczba stron", st.session_state.total_pages)

def handle_file_upload(uploaded_file):
    try:
        with st.spinner("Ładowanie pliku PDF..."):
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            st.session_state.pdf_doc = pdf_doc; st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.project_name = sanitize_filename(Path(uploaded_file.name).stem)
            st.session_state.total_pages = len(pdf_doc); st.session_state.current_page = 0
            st.session_state.extracted_pages = [None] * len(pdf_doc); st.session_state.processing_status = 'idle'
            st.session_state.next_batch_start_index = 0; st.session_state.meta_tags = {}
    except Exception as e:
        st.error(f"❌ Błąd ładowania pliku: {e}"); st.session_state.pdf_doc = None
    st.rerun()

def start_ai_processing():
    if not st.session_state.api_key: st.error("⚠️ Proszę podać klucz API OpenAI."); return
    st.session_state.processing_status = 'in_progress'; st.session_state.next_batch_start_index = 0
    st.session_state.extracted_pages = [None] * st.session_state.total_pages

def render_processing_status():
    if st.session_state.processing_status == 'idle': return
    processed_count = sum(1 for p in st.session_state.extracted_pages if p is not None)
    progress = processed_count / st.session_state.total_pages if st.session_state.total_pages > 0 else 0
    
    if st.session_state.processing_status == 'complete': st.success(f"✅ Przetwarzanie zakończone!")
    else: st.info(f"🔄 Przetwarzanie w toku... ({processed_count}/{st.session_state.total_pages})"); st.progress(progress)
    
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("💾 Zapisz postęp", use_container_width=True): save_project()
    articles = [p for p in st.session_state.extracted_pages if p and p.get('type') == 'artykuł']
    if articles:
        zip_data = [{'name': f"strona_{a['page_number']}.txt", 'content': a['raw_markdown'].encode('utf-8')} for a in articles if 'raw_markdown' in a]
        if zip_data: c2.download_button("📥 Pobierz artykuły", create_zip_archive(zip_data), f"{st.session_state.project_name}_artykuly.zip", "application/zip", use_container_width=True)

def render_navigation():
    if st.session_state.total_pages <= 1: return
    st.subheader("📖 Nawigacja")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("⬅️ Poprzednia", use_container_width=True, disabled=(st.session_state.current_page == 0)): st.session_state.current_page -= 1; st.rerun()
    c2.metric("Strona", f"{st.session_state.current_page + 1} / {st.session_state.total_pages}")
    if c3.button("Następna ➡️", use_container_width=True, disabled=(st.session_state.current_page >= st.session_state.total_pages - 1)): st.session_state.current_page += 1; st.rerun()
    
    new_page = st.slider("Przejdź do strony:", 1, st.session_state.total_pages, st.session_state.current_page + 1) - 1
    if new_page != st.session_state.current_page: st.session_state.current_page = new_page; st.rerun()

def render_page_content():
    st.divider()
    pdf_col, text_col = st.columns(2, gap="large")
    page_index = st.session_state.current_page

    with pdf_col:
        st.subheader(f"📄 Oryginał (Strona {page_index + 1})")
        image_data = render_page_as_image(st.session_state.pdf_doc, page_index)
        if image_data:
            st.image(image_data, use_container_width=True)
            images = extract_images_from_page(st.session_state.pdf_doc, page_index)
            if images:
                with st.expander(f"🖼️ Pokaż/ukryj {len(images)} obraz(y) z tej strony"):
                    for img in images: st.image(img['image'], caption=f"Obraz {img['index'] + 1}", use_container_width=True)
                img_zip = create_zip_archive([{'name': f"str_{page_index+1}_img_{i['index']}.{i['ext']}", 'content': i['image']} for i in images])
                st.download_button("Pobierz obrazy", img_zip, f"obrazy_strona_{page_index+1}.zip", "application/zip", use_container_width=True)
        else: st.error("Nie można wyświetlić podglądu strony.")

    with text_col:
        st.subheader("🤖 Tekst przetworzony przez AI")
        
        raw_text = st.session_state.pdf_doc.load_page(page_index).get_text("text")
        with st.expander("👁️ Pokaż surowy tekst wejściowy z tej strony"):
            st.text_area("Surowy tekst", raw_text, height=200, disabled=True, key=f"raw_text_{page_index}")
        
        page_result = st.session_state.extracted_pages[page_index]
        if page_result:
            page_type = page_result.get('type', 'nieznany')
            color = {"artykuł": "green", "reklama": "orange", "pominięta": "grey"}.get(page_type, "red")
            st.markdown(f"**Status:** <span style='color:{color}; text-transform:uppercase;'>**{page_type}**</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='page-text-wrapper'>{page_result.get('formatted_content', '')}</div>", unsafe_allow_html=True)
            action_cols = st.columns(2)
            if action_cols[0].button("🔄 Przetwórz ponownie (z kontekstem)", key=f"reroll_{page_index}", use_container_width=True):
                with st.spinner("Przetwarzanie strony z kontekstem..."):
                    prev_text = st.session_state.pdf_doc.load_page(page_index - 1).get_text("text") if page_index > 0 else ""
                    curr_text = st.session_state.pdf_doc.load_page(page_index).get_text("text")
                    next_text = st.session_state.pdf_doc.load_page(page_index + 1).get_text("text") if page_index < st.session_state.total_pages - 1 else ""
                    context_text = (f"KONTEKST (POPRZEDNIA STRONA):\n{prev_text}\n\n--- STRONA DOCELOWA ---\n{curr_text}\n\nKONTEKST (NASTĘPNA STRONA):\n{next_text}")
                    client = AsyncOpenAI(api_key=st.session_state.api_key)
                    new_result = asyncio.run(process_page_async(client, page_index + 1, context_text, st.session_state.model))
                    st.session_state.extracted_pages[page_index] = new_result
                st.rerun()
            if page_type == 'artykuł' and 'raw_markdown' in page_result:
                 if action_cols[1].button("✨ SEO: Generuj Meta Tagi", key=f"meta_{page_index}", use_container_width=True):
                    with st.spinner("Generowanie meta tagów..."):
                        client = AsyncOpenAI(api_key=st.session_state.api_key)
                        tags = asyncio.run(generate_meta_tags_async(client, page_result['raw_markdown'], st.session_state.model))
                        st.session_state.meta_tags[page_index] = tags
                    st.rerun()
            if page_index in st.session_state.meta_tags:
                tags = st.session_state.meta_tags[page_index]
                if "error" in tags: st.error(f"Błąd generowania meta tagów: {tags['error']}")
                else:
                    with st.expander("Wygenerowane Meta Tagi ✨", expanded=True):
                        st.text_input("Meta Title", value=tags.get("meta_title", ""), key=f"mt_{page_index}")
                        st.text_area("Meta Description", value=tags.get("meta_description", ""), key=f"md_{page_index}")
        else:
            st.info("⏳ Strona oczekuje na przetworzenie..." if st.session_state.processing_status == 'in_progress' else "Uruchom przetwarzanie w panelu bocznym.")

def main():
    st.title("🚀 Redaktor AI - Interaktywny Procesor PDF")
    render_sidebar()
    if not st.session_state.pdf_doc:
        st.info("👋 Witaj! Aby rozpocząć, wgraj plik PDF lub załaduj istniejący projekt z panelu bocznego."); return
    render_processing_status()
    if st.session_state.processing_status == 'in_progress' and st.session_state.next_batch_start_index < st.session_state.total_pages:
        asyncio.run(process_batch(st.session_state.next_batch_start_index))
        st.session_state.next_batch_start_index += BATCH_SIZE; st.rerun()
    elif st.session_state.processing_status == 'in_progress':
        st.session_state.processing_status = 'complete'; st.rerun()
    render_navigation(); render_page_content()

if __name__ == "__main__":
    main()
