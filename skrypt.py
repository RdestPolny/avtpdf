import streamlit as st
import fitz  # PyMuPDF
import json
import io
import zipfile
import re
from pathlib import Path
import asyncio
from openai import AsyncOpenAI

# =============================
# Konfiguracja i stałe
# =============================
INSTRUCTIONS_MD = """
### Witaj w Redaktorze AI! Oto krótki przewodnik, jak zacząć:

**Krok 1: Rozpocznij Projekt**
*   **Nowy plik PDF**: Użyj przycisku **"Wybierz plik PDF"** w panelu bocznym, aby wgrać swój dokument. Aplikacja automatycznie utworzy nowy projekt na podstawie nazwy pliku.
*   **Istniejący projekt**: Jeśli pracowałeś już nad plikiem, wybierz go z listy **"Wybierz istniejący projekt"** i kliknij **"Załaduj projekt"**. Następnie wgraj ten sam plik PDF, aby kontynuować pracę.

**Krok 2: Uruchom Przetwarzanie AI**
*   Gdy plik PDF jest załadowany, w panelu bocznym wybierz, czy chcesz przetworzyć **cały dokument**, czy tylko **zakres stron**.
*   Kliknij duży, niebieski przycisk **"🚀 Rozpocznij Przetwarzanie"**.
*   AI przeanalizuje wybrane strony, próbując zidentyfikować i sformatować artykuły. Możesz śledzić postęp na pasku postępu.

**Krok 3: Przeglądaj i Edytuj Wyniki**
*   Po zakończeniu przetwarzania (lub w jego trakcie) możesz nawigować między stronami za pomocą przycisków **"Poprzednia" / "Następna"** lub suwaka.
*   **Po lewej stronie** widzisz oryginalny wygląd strony z pliku PDF.
*   **Po prawej stronie** znajduje się tekst przetworzony przez AI.
*   **Co możesz zrobić na każdej stronie?**
    *   **🔄 Przetwórz ponownie (z kontekstem)**: Jeśli AI niepoprawnie zinterpretowało tekst, ten przycisk wyśle do AI bieżącą stronę wraz z tekstem ze stron sąsiednich, co często poprawia wynik.
    *   **✨ SEO: Generuj Meta Tagi**: Dla stron oznaczonych jako "ARTYKUŁ", możesz automatycznie wygenerować propozycje tytułu i opisu meta.
    *   **Pobierz obrazy**: Pobierz wszystkie grafiki z bieżącej strony w jednym pliku `.zip`.

**Krok 4: Zapisz i Eksportuj**
*   **💾 Zapisz postęp**: W dowolnym momencie możesz zapisać aktualny stan swojej pracy.
*   **📥 Pobierz artykuły**: Gdy będziesz zadowolony z wyników, kliknij ten przycisk, aby pobrać wszystkie przetworzone artykuły w formacie Markdown (`.txt`) spakowane do jednego pliku `.zip`.

Miłej pracy! 🚀
"""

st.set_page_config(
    layout="wide",
    page_title="Redaktor AI - Procesor PDF",
    page_icon="🚀",
)

# Wygląd
st.markdown(
    """
<style>
    .stApp { background-color: #F0F2F5; font-family: 'Segoe UI', sans-serif; }
    .main > div { background-color: #FFFFFF; padding: 2rem; border-radius: 15px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1); }
    .stButton>button { border-radius: 8px; border: 1px solid #E0E0E0; color: #333333; background-color: #FFFFFF; transition: all 0.2s ease-in-out; padding: 0.5rem 1rem; }
    .stButton>button:hover { border-color: #007BFF; color: #007BFF; box-shadow: 0 2px 5px 0 rgba(0, 123, 255, 0.2); }
    .stButton[data-testid="stButton-ProcessAI"] button { background-color: #007BFF; color: white; font-weight: bold; }
    .page-text-wrapper { padding: 1.5rem; border-radius: 8px; border: 1px solid #E8E8E8; background-color: #FAFAFA; min-height: 500px; max-height: 600px; overflow-y: auto; }
    .page-text-wrapper h4 { color: #007BFF; border-bottom: 2px solid #007BFF; padding-bottom: 8px; margin-bottom: 1rem; }
    .stMetric { text-align: center; }
    .error-box { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# Stan aplikacji
# =============================
PROJECTS_DIR = Path("pdf_processor_projects")
BATCH_SIZE = 10
MAX_RETRIES = 2
DEFAULT_MODEL = "gpt-4o-mini"


def init_session_state():
    defaults = {
        "processing_status": "idle",
        "pdf_doc": None,
        "current_page": 0,
        "total_pages": 0,
        "extracted_pages": [],
        "project_name": None,
        "next_batch_start_index": 0,
        "uploaded_filename": None,
        "api_key": st.secrets.get("openai", {}).get("api_key"),
        "model": DEFAULT_MODEL,
        "meta_tags": {},
        "project_loaded_and_waiting_for_pdf": False,
        "processing_mode": "all",
        "start_page": 1,
        "end_page": 1,
        "processing_end_page_index": 0,
        "article_page_groups_input": "",
        "article_groups": [],
        "next_article_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# =============================
# Narzędzia / utils
# =============================

def ensure_projects_dir() -> bool:
    try:
        PROJECTS_DIR.mkdir(exist_ok=True)
        return True
    except Exception as e:
        st.error(f"Nie można utworzyć katalogu projektów: {e}")
        return False


def get_existing_projects():
    if not ensure_projects_dir():
        return []
    return [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]


def sanitize_filename(name: str) -> str:
    if not name:
        return "unnamed_project"
    sanitized = re.sub(r"[\\/*?:\"<>|]", "_", str(name))
    return re.sub(r"_{2,}", "_", sanitized).strip("_") or "unnamed_project"


# =============================
# PDF helpers
# =============================

def render_page_as_image(pdf_doc, page_num):
    try:
        if not pdf_doc or not (0 <= page_num < len(pdf_doc)):
            return None
        page = pdf_doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        return pix.tobytes("png")
    except Exception as e:
        st.error(f"Błąd podczas renderowania strony {page_num + 1}: {e}")
        return None


def extract_images_from_page(pdf_doc, page_num):
    images = []
    if not pdf_doc or not (0 <= page_num < len(pdf_doc)):
        return images
    try:
        page = pdf_doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            if (
                base_image
                and base_image.get("width", 0) > 100
                and base_image.get("height", 0) > 100
            ):
                images.append(
                    {
                        "image": base_image["image"],
                        "ext": base_image["ext"],
                        "index": img_index,
                    }
                )
    except Exception as e:
        st.warning(
            f"Nie udało się wyekstraktować obrazów ze strony {page_num + 1}: {e}"
        )
    return images


def create_zip_archive(items):
    """items: list[{'name': str, 'content': bytes}]"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for it in items:
            zf.writestr(it["name"], it["content"])
    return buf.getvalue()


# =============================
# Parsowanie zakresów stron
# =============================

def parse_page_groups(input_text: str, total_pages: int):
    if not input_text:
        raise ValueError("Nie podano zakresów stron.")
    groups = []
    used = set()
    for line in re.split(r"[;\n]+", input_text):
        line = line.strip()
        if not line:
            continue
        pages = []
        for part in re.split(r"[;,]+", line):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                s, e = part.split("-", 1)
                if not s.isdigit() or not e.isdigit():
                    raise ValueError(f"Niepoprawny zakres stron: '{part}'.")
                s, e = int(s), int(e)
                if s > e:
                    raise ValueError(f"Zakres stron musi być rosnący: '{part}'.")
                if s < 1 or e > total_pages:
                    raise ValueError(
                        f"Zakres '{part}' wykracza poza liczbę stron dokumentu."
                    )
                pages.extend(range(s, e + 1))
            else:
                if not part.isdigit():
                    raise ValueError(f"Niepoprawny numer strony: '{part}'.")
                p = int(part)
                if p < 1 or p > total_pages:
                    raise ValueError(f"Strona '{p}' wykracza poza dokument.")
                pages.append(p)
        if not pages:
            continue
        pages = sorted(dict.fromkeys(pages))
        if any(p in used for p in pages):
            raise ValueError(f"Strony {pages} zostały już przypisane do innego artykułu.")
        used.update(pages)
        groups.append(pages)
    if not groups:
        raise ValueError("Nie znaleziono żadnych poprawnych zakresów stron.")
    return groups


# =============================
# Trwałość projektu
# =============================

def _state_to_save():
    excluded = {"pdf_doc", "project_loaded_and_waiting_for_pdf"}
    data = {k: v for k, v in st.session_state.items() if k not in excluded}
    data["extracted_pages"] = [p for p in st.session_state.extracted_pages if p is not None]
    return data


def save_project():
    if not st.session_state.project_name or not ensure_projects_dir():
        st.error("Nie można zapisać projektu: brak nazwy projektu.")
        return
    project_path = PROJECTS_DIR / st.session_state.project_name
    project_path.mkdir(exist_ok=True)
    try:
        with open(project_path / "project_state.json", "w", encoding="utf-8") as f:
            json.dump(_state_to_save(), f, indent=2, ensure_ascii=False)
        st.toast(
            f"✅ Projekt '{st.session_state.project_name}' został zapisany!", icon="💾"
        )
    except Exception as e:
        st.error(f"Błąd podczas zapisywania projektu: {e}")


def load_project(project_name: str):
    fpath = PROJECTS_DIR / project_name / "project_state.json"
    if not fpath.exists():
        st.error(f"Plik projektu '{project_name}' nie istnieje.")
        return
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k != "pdf_doc":
                st.session_state[k] = v
        total = st.session_state.get("total_pages", 0)
        st.session_state.extracted_pages = [None] * total
        for page_data in data.get("extracted_pages", []):
            pn = page_data.get("page_number")
            if pn and 1 <= pn <= total:
                st.session_state.extracted_pages[pn - 1] = page_data
        st.session_state.pdf_doc = None
        st.session_state.project_loaded_and_waiting_for_pdf = True
        st.success(
            f"✅ Załadowano projekt '{project_name}'. Wgraj powiązany plik PDF, aby kontynuować."
        )
    except Exception as e:
        st.error(f"Błąd podczas ładowania projektu: {e}")


# =============================
# Konwersje / formatowanie
# =============================

def markdown_to_html(text: str) -> str:
    text = text.replace("\n---\n", "\n<hr>\n")
    text = re.sub(r"^\s*# (.*?)\s*$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*## (.*?)\s*$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*### (.*?)\s*$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    paragraphs = text.split("\n\n")
    html = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith(("<h", "<hr")):
            html.append(para)
        else:
            html.append(f"<p>{para.replace(chr(10), '<br>')}</p>")
    return "".join(html)


def _clean_raw_text_for_prompt(text: str) -> str:
    """Minimalne oczyszczenie: usunięcie nadmiarowych pustych linii i spacji."""
    # Usuń rozstrzelone daty typu 'c z e r w i e c  2 0 2 5'
    text = re.sub(r"(?:\b\w\s){3,}\d{4}", lambda m: m.group(0).replace(" ", ""), text)
    # Zredukuj wielokrotne puste linie
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Uporządkuj spacje
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# =============================
# OpenAI helpers (wyłącznie JSON)
# =============================

def _strip_code_fences(s: str) -> str:
    cs = s.strip()
    if cs.startswith("```json"):
        cs = cs[len("```json"):].strip()
    if cs.startswith("```"):
        cs = cs[len("```"):].strip()
    if cs.endswith("```"):
        cs = cs[:-3].strip()
    return cs


def _extract_first_json_object(s: str) -> str:
    """Wyciąga pierwszy komplet obiektu JSON skanując nawiasy klamrowe."""
    s = _strip_code_fences(s)
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s  # fallback


async def _call_json_chat(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int = 4096, temperature: float = 0.1):
    last_error = None
    raw_content = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_content = resp.choices[0].message.content or ""
            cleaned = _extract_first_json_object(raw_content)
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)
            continue
        except Exception as e:
            last_error = e
            break
    return {"__error__": f"Błąd parsowania po {MAX_RETRIES + 1} próbach: {last_error}", "__raw__": raw_content}


# =============================
# Prompty
# =============================
PAGE_PROMPT_TMPL = (
    "Jesteś precyzyjnym asystentem redakcyjnym. Twoim celem jest przekształcenie surowego tekstu w czytelny, dobrze "
    "zorganizowany artykuł internetowy. ZASADA NADRZĘDNA: WIERNOŚĆ TREŚCI, ELASTYCZNOŚĆ FORMY. - **Nie zmieniaj "
    "oryginalnych sformułowań ani nie parafrazuj tekstu.** Twoim zadaniem jest przenieść treść 1:1. - Twoja rola "
    "polega na **dodawaniu elementów strukturalnych** (nagłówki, pogrubienia, podział na akapity) w celu poprawy "
    "czytelności. INSTRUKCJE SPECJALNE: Oczyszczanie i Kontekst. 1.  **Ignorowanie Szumu**: Musisz zidentyfikować i "
    "całkowicie pominąć w finalnym tekście następujące elementy, jeśli występują na początku lub końcu: - Numery stron "
    "(np. '6', '12'). - Rozstrzelone daty (np. 'c z e r w i e c  2 0 2 5'). 2.  **Wykorzystanie Kategorii**: - Na "
    "początku tekstu możesz znaleźć etykietę, np. 'od redakcji', 'NEWS FLASH', 'WYWIAD MIESIĄCA'. - Użyj tej etykiety "
    "jako **kontekstu** do zrozumienia intencji tekstu, ale **nie umieszczaj jej w sformatowanym artykule**. DOZWOLONE "
    "MODYFIKACJE STRUKTURALNE: 1.  **Tytuł Główny (`# Tytuł`)**: - **ZNAJDŹ go w tekście**. To często krótka linia bez "
    "kropki na końcu. 2.  **Śródtytuły (`## Śródtytuł`) - Twój kluczowy obowiązek**: - Celem jest przekształcenie 'ściany "
    "tekstu' w czytelny artykuł. - Gdy tekst zawiera kilka następujących po sobie akapitów omawiających różne przykłady, "
    "technologie lub firmy, **MUSISZ** je rozdzielić trafnymi, zwięzłymi śródtytułami. 3.  **Pogrubienia (`**tekst**`)**: "
    "- Stosuj je, by wyróżnić **kluczowe terminy, nazwy własne i frazy ważne dla SEO**. 4.  **Podział na sekcje**: - Jeśli "
    "tekst na stronie wyraźnie zawiera **dwa lub więcej niepowiązanych ze sobą tematów**, oddziel je linią horyzontalną "
    "(`---`). WYMAGANIA KRYTYCZNE: - Twoja odpowiedź musi być **WYŁĄCZNIE i BEZWZGLĘDNIE** poprawnym obiektem JSON. "
    "FORMAT ODPOWIEDZI: {\"type\": \"ARTYKUŁ\" lub \"REKLAMA\", \"formatted_text\": \"Sformatowany tekst w markdown. Jeśli jest wiele artykułów, oddziel je '---'.\"}} "
    "TEKST DO PRZETWORZENIA: --- {raw_text} ---"
)

ARTICLE_PROMPT_TMPL = (
    "Jesteś precyzyjnym asystentem redakcyjnym. Poniżej otrzymujesz surowy tekst artykułu rozłożonego na kilka stron. "
    "Twoim zadaniem jest przygotowanie spójnego, kompletnego artykułu w formacie markdown zgodnie z zasadami: ZASADA "
    "NADRZĘDNA: WIERNOŚĆ TREŚCI, ELASTYCZNOŚĆ FORMY. - **Nie zmieniaj oryginalnych sformułowań ani nie parafrazuj tekstu.** "
    "Twoim zadaniem jest przenieść treść 1:1. - Twoja rola polega na **dodawaniu elementów strukturalnych** (nagłówki, "
    "pogrubienia, podział na akapity) w celu poprawy czytelności. INSTRUKCJE SPECJALNE: Oczyszczanie i Kontekst. 1.  "
    "**Ignorowanie Szumu**: Musisz zidentyfikować i całkowicie pominąć w finalnym tekście następujące elementy, jeśli "
    "występują na początku lub końcu: - Numery stron. - Rozstrzelone daty. 2.  **Wykorzystanie Kategorii**: - Na początku "
    "tekstu możesz znaleźć etykietę, np. 'od redakcji', 'NEWS FLASH', 'WYWIAD MIESIĄCA'. - Użyj tej etykiety jako "
    "**kontekstu** do zrozumienia intencji tekstu, ale **nie umieszczaj jej w sformatowanym artykule**. DOZWOLONE "
    "MODYFIKACJE STRUKTURALNE: 1.  **Tytuł Główny (`# Tytuł`)**: - **ZNAJDŹ go w tekście**. 2.  **Śródtytuły (`## Śródtytuł`)**: "
    "- Rozbij długie sekcje na logiczne części. 3.  **Pogrubienia (`**tekst**`)**: - Wyróżnij kluczowe informacje. 4.  "
    "**Podział na sekcje**: - Jeśli tekst zawiera różne tematy, oddziel je linią horyzontalną (`---`). WYMAGANIA KRYTYCZNE: - "
    "Odpowiedź musi być **WYŁĄCZNIE** poprawnym obiektem JSON w formacie: {\"type\": \"ARTYKUŁ\" lub \"REKLAMA\", "
    "\"formatted_text\": \"Sformatowany tekst w markdown.\"}}. TEKST DO PRZETWORZENIA: --- {raw_text} ---"
)

META_PROMPT_TMPL = (
    "Jesteś ekspertem SEO. Na podstawie poniższego tekstu artykułu, wygeneruj chwytliwy meta title i zwięzły meta "
    "description. WYMAGANIA: - Meta title: max 60 znaków. - Meta description: max 160 znaków. - Odpowiedź zwróć jako "
    "obiekt JSON. FORMAT ODPOWIEDZI: {\"meta_title\": \"Tytuł meta\", \"meta_description\": \"Opis meta.\"}} "
    "TEKST ARTYKUŁU: --- {raw_text} ---"
)


# =============================
# Funkcje AI (strony i artykuły)
# =============================
async def process_page_async(client: AsyncOpenAI, page_num: int, raw_text: str, model: str):
    page_data = {"page_number": page_num}

    # Zbyt mało treści – pomiń
    if len(raw_text.split()) < 20:
        page_data["type"] = "pominięta"
        page_data["formatted_content"] = "<i>Strona zawiera zbyt mało tekstu.</i>"
        return page_data

    prompt = PAGE_PROMPT_TMPL.format(raw_text=_clean_raw_text_for_prompt(raw_text))

    result = await _call_json_chat(client, model, prompt, max_tokens=2048, temperature=0.1)

    if "__error__" in result:
        page_data["type"] = "błąd"
        page_data["formatted_content"] = (
            f"""<div class=\"error-box\"><strong>Błąd parsowania.</strong><br><i>{result['__error__']}</i>"""
            + (f"<br><details><summary>Pokaż ostatnią surową odpowiedź</summary><pre>{result.get('__raw__','')}</pre></details></div>")
        )
        return page_data

    page_type = result.get("type", "nieznany").lower()
    formatted_text = result.get("formatted_text", "")

    page_data["type"] = page_type
    if page_type == "artykuł":
        page_data["formatted_content"] = markdown_to_html(formatted_text)
        page_data["raw_markdown"] = formatted_text
    else:
        page_data["formatted_content"] = (
            f"<i>Zidentyfikowano jako: <strong>{page_type.upper()}</strong>.</i>"
        )
    return page_data


async def process_article_group_async(client: AsyncOpenAI, page_numbers: list[int], raw_text: str, model: str):
    prompt = ARTICLE_PROMPT_TMPL.format(raw_text=_clean_raw_text_for_prompt(raw_text))

    result = await _call_json_chat(client, model, prompt, max_tokens=4096, temperature=0.1)

    content = {
        "page_numbers": page_numbers,
    }

    if "__error__" in result:
        content.update(
            {
                "type": "błąd",
                "formatted_content": (
                    f"""<div class='error-box'><strong>Błąd parsowania.</strong><br><i>{result['__error__']}</i>"""
                    + (f"<br><details><summary>Pokaż ostatnią surową odpowiedź</summary><pre>{result.get('__raw__','')}</pre></details></div>")
                ),
            }
        )
        return content

    art_type = result.get("type", "nieznany").lower()
    formatted_text = result.get("formatted_text", "")

    if art_type == "artykuł":
        content.update(
            {
                "type": art_type,
                "formatted_content": markdown_to_html(formatted_text),
                "raw_markdown": formatted_text,
            }
        )
    else:
        content.update(
            {
                "type": art_type,
                "formatted_content": f"<i>Zidentyfikowano jako: <strong>{art_type.upper()}</strong>.</i>",
            }
        )
    return content


async def generate_meta_tags_async(client: AsyncOpenAI, article_text: str, model: str):
    prompt = META_PROMPT_TMPL.format(raw_text=article_text[:4000])
    result = await _call_json_chat(client, model, prompt, max_tokens=200, temperature=0.5)
    if "__error__" in result:
        return {"error": result["__error__"]}
    return result


# =============================
# Przetwarzanie wsadowe
# =============================
async def process_batch(start_index: int):
    client = AsyncOpenAI(api_key=st.session_state.api_key)
    processing_limit = st.session_state.processing_end_page_index + 1
    end_index = min(start_index + BATCH_SIZE, processing_limit)
    tasks = []
    for i in range(start_index, end_index):
        if st.session_state.pdf_doc:
            raw_text = st.session_state.pdf_doc.load_page(i).get_text("text")
            tasks.append(process_page_async(client, i + 1, raw_text, st.session_state.model))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        page_index = start_index + i
        if isinstance(result, Exception):
            st.session_state.extracted_pages[page_index] = {
                "page_number": page_index + 1,
                "type": "błąd",
                "formatted_content": f"Błąd: {result}",
            }
        else:
            st.session_state.extracted_pages[page_index] = result


# =============================
# UI: Sidebar i obsługa pliku
# =============================

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Konfiguracja Projektu")
        st.subheader("📂 Projekt")
        projects = get_existing_projects()
        selected_project = st.selectbox("Wybierz istniejący projekt", ["Nowy projekt"] + projects)
        if st.button("Załaduj projekt", disabled=(selected_project == "Nowy projekt")):
            load_project(selected_project)
            st.rerun()

        st.divider()
        st.subheader("📄 Plik PDF")
        uploaded_file = st.file_uploader("Wybierz plik PDF", type="pdf")

        # Logika obsługi pliku
        if uploaded_file and st.session_state.project_loaded_and_waiting_for_pdf:
            handle_file_upload(uploaded_file)
        elif uploaded_file and uploaded_file.name != st.session_state.get("uploaded_filename"):
            handle_file_upload(uploaded_file)

        if st.session_state.pdf_doc:
            st.divider()
            st.subheader("🤖 Opcje Przetwarzania")
            st.radio(
                "Wybierz tryb:",
                ("all", "range", "article"),
                captions=[
                    "Przetwórz cały dokument",
                    "Przetwórz zakres stron",
                    "Przetwórz wielostronicowy artykuł",
                ],
                key="processing_mode",
                horizontal=True,
            )
            if st.session_state.processing_mode == "range":
                c1, c2 = st.columns(2)
                c1.number_input(
                    "Od strony",
                    min_value=1,
                    max_value=st.session_state.total_pages,
                    key="start_page",
                )
                c2.number_input(
                    "Do strony",
                    min_value=st.session_state.start_page,
                    max_value=st.session_state.total_pages,
                    key="end_page",
                )
            elif st.session_state.processing_mode == "article":
                st.info(
                    "Podaj grupy stron należących do jednego artykułu. Każdą grupę oddziel średnikiem lub nową linią, np. `11-15; 5,6`."
                )
                st.text_area(
                    "Zakresy stron artykułów",
                    key="article_page_groups_input",
                    placeholder="11-15\n5,6",
                )

            st.divider()
            processing_disabled = (
                st.session_state.processing_status == "in_progress"
                or not st.session_state.api_key
            )
            btn_text = (
                "🔄 Przetwarzanie..."
                if st.session_state.processing_status == "in_progress"
                else "🚀 Rozpocznij Przetwarzanie"
            )
            if st.button(
                btn_text,
                use_container_width=True,
                type="primary",
                disabled=processing_disabled,
                key="stButton-ProcessAI",
            ):
                start_ai_processing()
                st.rerun()

        if st.session_state.project_name:
            st.divider()
            st.info(f"**Projekt:** `{st.session_state.project_name}`")
            st.metric("Liczba stron", st.session_state.total_pages)


def handle_file_upload(uploaded_file):
    try:
        with st.spinner("Ładowanie pliku PDF..."):
            pdf_bytes = uploaded_file.read()
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            if st.session_state.project_loaded_and_waiting_for_pdf:
                if len(pdf_doc) != st.session_state.total_pages:
                    st.error(
                        f"Błąd: Wgrany PDF ma {len(pdf_doc)} stron, a projekt oczekuje {st.session_state.total_pages}. Wgraj właściwy plik."
                    )
                    return
                st.session_state.pdf_doc = pdf_doc
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.project_loaded_and_waiting_for_pdf = False
                st.success("✅ Plik PDF pomyślnie dopasowany do projektu.")
            else:
                st.session_state.pdf_doc = pdf_doc
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.project_name = sanitize_filename(Path(uploaded_file.name).stem)
                st.session_state.total_pages = len(pdf_doc)
                st.session_state.current_page = 0
                st.session_state.extracted_pages = [None] * len(pdf_doc)
                st.session_state.processing_status = "idle"
                st.session_state.next_batch_start_index = 0
                st.session_state.meta_tags = {}
                st.session_state.end_page = len(pdf_doc)
    except Exception as e:
        st.error(f"❌ Błąd ładowania pliku: {e}")
        st.session_state.pdf_doc = None
    st.rerun()


# =============================
# Sterowanie przetwarzaniem
# =============================

def start_ai_processing():
    if not st.session_state.api_key:
        st.error("⚠️ Klucz API OpenAI nie jest skonfigurowany w sekretach.")
        return

    mode = st.session_state.processing_mode

    if mode == "article":
        try:
            groups = parse_page_groups(
                st.session_state.article_page_groups_input, st.session_state.total_pages
            )
        except ValueError as e:
            st.error(str(e))
            return

        # Wyczyść wyniki dla stron wchodzących w skład zgłoszonych artykułów
        for group in groups:
            for p in group:
                st.session_state.extracted_pages[p - 1] = None

        st.session_state.article_groups = groups
        st.session_state.next_article_index = 0
        st.session_state.processing_status = "in_progress"
        return

    # tryby 'all' lub 'range'
    if mode == "all":
        start_idx, end_idx = 0, st.session_state.total_pages - 1
    else:
        start_idx, end_idx = (
            st.session_state.start_page - 1,
            st.session_state.end_page - 1,
        )
        if start_idx > end_idx:
            st.error("Strona początkowa nie może być większa niż końcowa.")
            return

    for i in range(start_idx, end_idx + 1):
        st.session_state.extracted_pages[i] = None

    st.session_state.processing_status = "in_progress"
    st.session_state.next_batch_start_index = start_idx
    st.session_state.processing_end_page_index = end_idx


def assign_article_result_to_pages(article_result: dict):
    """Rozlej rezultat artykułu na wszystkie strony w grupie.
    Pierwsza strona w grupie jest liderem (is_group_lead=True)."""
    pages = article_result.get("page_numbers", [])
    for page in pages:
        idx = page - 1
        if 0 <= idx < len(st.session_state.extracted_pages):
            entry = {k: v for k, v in article_result.items() if k != "page_numbers"}
            entry["page_number"] = page
            entry["group_pages"] = pages
            entry["is_group_lead"] = page == pages[0]
            st.session_state.extracted_pages[idx] = entry


# =============================
# UI: Pasek statusu / Nawigacja / Zawartość strony
# =============================

def render_processing_status():
    if st.session_state.processing_status == "idle":
        return

    processed_count = sum(1 for p in st.session_state.extracted_pages if p is not None)
    progress = (
        processed_count / st.session_state.total_pages
        if st.session_state.total_pages > 0
        else 0
    )

    if st.session_state.processing_status == "complete":
        st.success("✅ Przetwarzanie zakończone!")
    else:
        st.info(
            f"🔄 Przetwarzanie w toku... (Ukończono {processed_count}/{st.session_state.total_pages} stron dokumentu)"
        )
        st.progress(progress)

    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("💾 Zapisz postęp", use_container_width=True):
        save_project()

    # Zbieraj tylko liderów grup artykułów
    articles = [
        p
        for p in st.session_state.extracted_pages
        if p and p.get("type") == "artykuł" and p.get("is_group_lead", True)
    ]
    if articles:
        zip_items = []
        for a in articles:
            if "raw_markdown" in a:
                if gp := a.get("group_pages"):
                    label = (
                        f"artykul_strony_{gp[0]}-{gp[-1]}" if len(gp) > 1 else f"strona_{a['page_number']}"
                    )
                else:
                    label = f"strona_{a['page_number']}"
                zip_items.append(
                    {"name": f"{label}.txt", "content": a["raw_markdown"].encode("utf-8")}
                )
        if zip_items:
            c2.download_button(
                "📥 Pobierz artykuły",
                create_zip_archive(zip_items),
                f"{st.session_state.project_name}_artykuly.zip",
                "application/zip",
                use_container_width=True,
            )


def render_navigation():
    if st.session_state.total_pages <= 1:
        return
    st.subheader("📖 Nawigacja")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("⬅️ Poprzednia", use_container_width=True, disabled=(st.session_state.current_page == 0)):
        st.session_state.current_page -= 1
        st.rerun()
    c2.metric("Strona", f"{st.session_state.current_page + 1} / {st.session_state.total_pages}")
    if c3.button(
        "Następna ➡️",
        use_container_width=True,
        disabled=(st.session_state.current_page >= st.session_state.total_pages - 1),
    ):
        st.session_state.current_page += 1
        st.rerun()
    new_page = (
        st.slider(
            "Przejdź do strony:", 1, st.session_state.total_pages, st.session_state.current_page + 1
        )
        - 1
    )
    if new_page != st.session_state.current_page:
        st.session_state.current_page = new_page
        st.rerun()


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
                    for img in images:
                        st.image(img["image"], caption=f"Obraz {img['index'] + 1}", use_container_width=True)
                img_zip = create_zip_archive(
                    [
                        {
                            "name": f"str_{page_index+1}_img_{i['index']}.{i['ext']}",
                            "content": i["image"],
                        }
                        for i in images
                    ]
                )
                st.download_button(
                    "Pobierz obrazy",
                    img_zip,
                    f"obrazy_strona_{page_index+1}.zip",
                    "application/zip",
                    use_container_width=True,
                )
        else:
            st.error("Nie można wyświetlić podglądu strony.")

    with text_col:
        st.subheader("🤖 Tekst przetworzony przez AI")
        raw_text = st.session_state.pdf_doc.load_page(page_index).get_text("text")
        with st.expander("👁️ Pokaż surowy tekst wejściowy z tej strony"):
            st.text_area(
                "Surowy tekst", raw_text, height=200, disabled=True, key=f"raw_text_{page_index}"
            )

        page_result = st.session_state.extracted_pages[page_index]
        if page_result:
            page_type = page_result.get("type", "nieznany")
            color = {"artykuł": "green", "reklama": "orange", "pominięta": "grey"}.get(
                page_type, "red"
            )
            st.markdown(
                f"**Status:** <span style='color:{color}; text-transform:uppercase;'>**{page_type}**</span>",
                unsafe_allow_html=True,
            )
            group_pages = page_result.get("group_pages", [])
            if group_pages and len(group_pages) > 1:
                st.info(
                    f"Ten artykuł obejmuje strony: {', '.join(str(p) for p in group_pages)}."
                )
            st.markdown(
                f"<div class='page-text-wrapper'>{page_result.get('formatted_content', '')}</div>",
                unsafe_allow_html=True,
            )

            action_cols = st.columns(2)
            if action_cols[0].button(
                "🔄 Przetwórz ponownie (z kontekstem)",
                key=f"reroll_{page_index}",
                use_container_width=True,
            ):
                with st.spinner("Przetwarzanie strony z kontekstem..."):
                    prev_text = (
                        st.session_state.pdf_doc.load_page(page_index - 1).get_text("text")
                        if page_index > 0
                        else ""
                    )
                    curr_text = st.session_state.pdf_doc.load_page(page_index).get_text("text")
                    next_text = (
                        st.session_state.pdf_doc.load_page(page_index + 1).get_text("text")
                        if page_index < st.session_state.total_pages - 1
                        else ""
                    )
                    context_text = (
                        f"KONTEKST (POPRZEDNIA STRONA):\n{prev_text}\n\n--- STRONA DOCELOWA ---\n{curr_text}\n\nKONTEKST (NASTĘPNA STRONA):\n{next_text}"
                    )
                    client = AsyncOpenAI(api_key=st.session_state.api_key)
                    new_result = asyncio.run(
                        process_page_async(
                            client, page_index + 1, context_text, st.session_state.model
                        )
                    )
                    st.session_state.extracted_pages[page_index] = new_result
                st.rerun()

            allow_meta = (
                page_type == "artykuł"
                and "raw_markdown" in page_result
                and page_result.get("is_group_lead", True)
            )
            if allow_meta:
                if action_cols[1].button(
                    "✨ SEO: Generuj Meta Tagi",
                    key=f"meta_{page_index}",
                    use_container_width=True,
                ):
                    with st.spinner("Generowanie meta tagów..."):
                        client = AsyncOpenAI(api_key=st.session_state.api_key)
                        tags = asyncio.run(
                            generate_meta_tags_async(
                                client, page_result["raw_markdown"], st.session_state.model
                            )
                        )
                        st.session_state.meta_tags[page_index] = tags
                    st.rerun()
            elif page_type == "artykuł":
                action_cols[1].button(
                    "✨ SEO: Generuj Meta Tagi",
                    key=f"meta_{page_index}",
                    use_container_width=True,
                    disabled=True,
                )

            if page_index in st.session_state.meta_tags:
                tags = st.session_state.meta_tags[page_index]
                if "error" in tags:
                    st.error(f"Błąd generowania meta tagów: {tags['error']}")
                else:
                    with st.expander("Wygenerowane Meta Tagi ✨", expanded=True):
                        st.text_input(
                            "Meta Title", value=tags.get("meta_title", ""), key=f"mt_{page_index}"
                        )
                        st.text_area(
                            "Meta Description",
                            value=tags.get("meta_description", ""),
                            key=f"md_{page_index}",
                        )
        else:
            st.info(
                "⏳ Strona oczekuje na przetworzenie..."
                if st.session_state.processing_status == "in_progress"
                else "Uruchom przetwarzanie w panelu bocznym."
            )


# =============================
# Główna pętla aplikacji
# =============================

def main():
    st.title("🚀 Redaktor AI - Interaktywny Procesor PDF")

    if not st.session_state.api_key:
        st.error("Brak klucza API OpenAI!")
        st.info("""Proszę skonfiguruj swój klucz API w pliku `.streamlit/secrets.toml`.""")
        st.code("[openai]\napi_key = \"sk-...\"", language="toml")
        st.stop()

    render_sidebar()

    if not st.session_state.pdf_doc:
        if not st.session_state.project_loaded_and_waiting_for_pdf:
            st.info(
                "👋 Witaj! Aby rozpocząć, wgraj plik PDF lub załaduj istniejący projekt z panelu bocznego."
            )
            with st.expander("📖 Jak korzystać z aplikacji? Kliknij, aby rozwinąć instrukcję"):
                st.markdown(INSTRUCTIONS_MD, unsafe_allow_html=True)
        return

    render_processing_status()

    # Silnik przetwarzania
    if st.session_state.processing_status == "in_progress":
        if st.session_state.processing_mode == "article":
            # Przetwarzaj po jednym artykule na 'tick'
            if st.session_state.next_article_index < len(st.session_state.article_groups):
                article_pages = st.session_state.article_groups[st.session_state.next_article_index]
                client = AsyncOpenAI(api_key=st.session_state.api_key)

                # Zbuduj surowy tekst łącząc strony; każda strona wyraźnie oznaczona
                combined = []
                for p in article_pages:
                    idx = p - 1
                    if st.session_state.pdf_doc and 0 <= idx < st.session_state.total_pages:
                        t = st.session_state.pdf_doc.load_page(idx).get_text("text")
                        combined.append(f"--- STRONA {p} ---\n{t.strip()}\n")
                raw_joined = "\n".join(combined)

                # Wywołaj model JEDEN RAZ dla całego artykułu
                article_result = asyncio.run(
                    process_article_group_async(
                        client, article_pages, raw_joined, st.session_state.model
                    )
                )

                # Rozlej wynik na strony zakresu
                assign_article_result_to_pages(article_result)

                # Następna grupa lub zakończenie
                st.session_state.next_article_index += 1
                st.rerun()
            else:
                st.session_state.processing_status = "complete"
                st.rerun()
        else:
            # tryby 'all' i 'range' idą wsadowo po BATCH_SIZE
            if st.session_state.next_batch_start_index <= st.session_state.processing_end_page_index:
                asyncio.run(process_batch(st.session_state.next_batch_start_index))
                st.session_state.next_batch_start_index += BATCH_SIZE
                st.rerun()
            else:
                st.session_state.processing_status = "complete"
                st.rerun()

    render_navigation()
    render_page_content()


if __name__ == "__main__":
    main()
