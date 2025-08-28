from __future__ import annotations
import io
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Optional

import time
import requests
import pandas as pd
import streamlit as st
from PIL import Image

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "sqlcoder")  # change if needed

try:
    import ollama  # type: ignore
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

try:
    from ultralytics import YOLO  # type: ignore
    _ULTRALYTICS_OK = True
except Exception:
    _ULTRALYTICS_OK = False

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


# ---------------- Image Classification ----------------
@dataclass
class Detection:
    label: str
    conf: float


class VehicleClassifier:
    """
    Klasyfikator pojazdów:
    - w trybie "yolo" używa modelu YOLO z biblioteki ultralytics,
    """

    def __init__(self, backend: str = "yolo", yolo_weights: str = "yolo11m.pt"):
        self.backend = backend
        self.yolo_weights = yolo_weights
        self._yolo_model = None
        if backend == "yolo" and _ULTRALYTICS_OK:
            try:
                self._yolo_model = YOLO(yolo_weights)
            except Exception as e:
                st.warning(f"Failed to load YOLO weights '{yolo_weights}': {e}")
                self._yolo_model = None

    def predict(self, image: Image.Image) -> List[Detection]:
        """Zwraca listę wykrytych pojazdów (etykieta + confidence) dla podanego obrazu."""
        if self.backend == "yolo":
            return self._predict_yolo(image)
        else:
            return []

    def _predict_yolo(self, image: Image.Image) -> List[Detection]:
        """Uruchamia YOLO na obrazie i zwraca pojazdy z VEHICLE_CLASSES."""
        if not _ULTRALYTICS_OK:
            st.info("Ultralytics not installed. Install with: pip install ultralytics")
            return []

        try:
            if self._yolo_model is None:
                self._yolo_model = YOLO(self.yolo_weights)

            # Upewniamy się, że obraz jest w RGB
            img_rgb = image.convert("RGB")

            # Przekazujemy obraz PIL bezpośrednio
            results = self._yolo_model.predict(source=img_rgb, verbose=False)

            out: List[Detection] = []
            for r in results:
                names = r.names
                for b in r.boxes:
                    cls_id = int(b.cls)
                    lbl = names.get(cls_id, str(cls_id))
                    if lbl.lower() in VEHICLE_CLASSES:
                        conf = float(b.conf)
                        out.append(Detection(label=lbl, conf=conf))
            return out

        except Exception as e:
            st.error(f"YOLO inference failed: {e}")
            return []


# ---------------- NL → SQL Agent ----------------
SQL_ASSISTANT_PROMPT = (
    "You are an SQL assistant. Generate valid SQLite SELECT queries only, strictly using the given schema.\n"
    "Rules:\n"
    "- Do NOT assume any columns exist that are not explicitly listed in the schema.\n"
    "- Use only the tables and columns in the schema to connect entities. If no direct column exists, find an intermediate table to join them.\n"
    "- Do not create imaginary relationships between tables.\n"
    "- Return ONLY one SQL block without explanations.\n"
    "- Always assign an alias when a table is used in FROM or JOIN, and use it consistently throughout, including in subqueries.\n"
    "- Do not mix table names with aliases.\n"
    "- Use explicit JOINs to connect tables instead of subqueries in ON clauses.\n"
    "- Always reference columns using the alias, never the raw table name.\n"
    "- Use only tables/columns present in the schema.\n"
    "- Do NOT assume columns or relationships that are not explicitly defined in the schema.\n"
    "- If a direct relationship between tables does not exist, look for intermediate tables in the schema to connect them.\n"
    "- Always validate that any column used in SELECT, WHERE, JOIN, or ON exists in the given schema.\n"
    "- Do NOT modify data (SELECT only).\n"
    "Schema:\n{SCHEMA}\n\nQuestion:\n{QUESTION}\n"
    "IMPORTANT: Always include 'vehicle_id' in your SELECT clause whenever selecting vehicle information, even if the user did not explicitly ask for it.\n"
)

FORBIDDEN_SQL = re.compile(r"\b(insert|update|delete|drop|alter|attach|pragma|create|replace|vacuum|reindex)\b", re.IGNORECASE)

def compact_schema_from_sqlite(conn: sqlite3.Connection) -> str:
    """Buduje krótki opis schematu bazy w formacie: Table X (col type, ...); – używane do promptu LLM."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    lines = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({t});")
        cols = [f"{c[1]} {c[2]}" for c in cur.fetchall()]  # name, type
        line = f"Table {t} (" + ", ".join(cols) + ");"
        lines.append(line)
    return "\n".join(lines)

def schema_dict_from_sqlite(conn: sqlite3.Connection) -> dict:
    """Zwraca słownik {nazwa_tabeli: [lista_kolumn]} dla całej bazy SQLite."""
    schema = {}
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for (table_name,) in cursor.fetchall():
        c2 = conn.execute(f"PRAGMA table_info({table_name});")
        cols = [row[1] for row in c2.fetchall()]
        schema[table_name] = cols
    return schema

def generate_and_test_sql(question: str, schema_text: str, conn: sqlite3.Connection, max_retries: int = 3) -> str:
    """
    Generuje zapytanie SQL przy użyciu modelu LLM (ollama).
    - Buduje prompt z pytaniem + schematem,
    - Próbnie uruchamia zapytanie (safe_execute_sql),
    - Podejmuje kilka prób z różnymi temperaturami,
    - Jeśli żadna próba się nie uda, zwraca ostatnie wygenerowane SQL z dopisanym błędem.
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Python package 'ollama' not installed. Run: pip install ollama")

    default_temp = 0.7
    temp_steps = [default_temp, 0.9, 1.0]

    last_error = None
    text = ""  # na wypadek gdyby nic nie zostało wygenerowane

    for attempt in range(max_retries):
        # --- generacja SQL ---
        prompt = SQL_ASSISTANT_PROMPT.format(SCHEMA=schema_text, QUESTION=question)
        temp = temp_steps[attempt] if attempt < len(temp_steps) else default_temp

        resp = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": temp,
                "num_ctx": 4096,
                "seed": int(time.time()) + attempt
            },
        )

        text = resp["message"]["content"].strip()

        # Wyciągnięcie SQL z bloku ```sql ... ```
        m = re.search(r"```(?:sql)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()

        # --- próba uruchomienia ---
        try:
            _ = safe_execute_sql(conn, text)  # sprawdzamy czy działa
            return text  # jeśli działa → zwracamy SQL
        except Exception as e:
            last_error = str(e)
            # nadal kontynuujemy kolejne próby

    # po max_retries zwracamy ostatnio wygenerowany kod, nawet jeśli jest błędny
    if last_error:
        text = f"{text}\n-- !!! Nieudane wykonanie: {last_error}"
    return text



# --- SQL validation helpers ---
ALIAS_CLAUSE = re.compile(r'\b(?:from|join)\s+([a-zA-Z_][\w]*)\s*(?:as\s+)?([a-zA-Z_][\w]*)?', re.IGNORECASE)
COLREF      = re.compile(r'([a-zA-Z_][\w]*)\s*\.\s*([a-zA-Z_][\w]*)')

def extract_table_aliases(sql: str) -> dict[str, str]:
    """Wyszukuje w SQL aliasy tabel i zwraca mapę {alias: tabela}."""
    aliases: dict[str, str] = {}
    for table, alias in ALIAS_CLAUSE.findall(sql):
        t = table.strip()
        a = (alias or table).strip()
        aliases[a] = t
    return aliases

def extract_column_refs(sql: str) -> list[tuple[str, str]]:
    """Zwraca wszystkie referencje kolumn w postaci [(alias, kolumna), ...]."""
    return [(a.strip(), c.strip()) for a, c in COLREF.findall(sql)]

def validate_sql(sql: str, schema: dict[str, list[str] | set[str]]) -> None:
    """
    Waliduje zapytanie SQL:
    - sprawdza czy to SELECT,
    - weryfikuje czy użyte tabele i kolumny istnieją w schemacie,
    - pilnuje by nie mieszać aliasów z pełnymi nazwami tabel.
    Rzuca ValueError, jeśli SQL jest niepoprawny.
    """
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    # schema -> lower-case for niezależność od wielkości liter
    schema_lc: dict[str, set[str]] = {t.lower(): {c.lower() for c in cols} for t, cols in schema.items()}

    # aliases
    alias_map = extract_table_aliases(sql)
    alias_map_lc = {a.lower(): t.lower() for a, t in alias_map.items()}

    # 1) sprawdź czy tabele istnieją
    missing_tables = [t for t in set(alias_map_lc.values()) if t not in schema_lc]
    if missing_tables:
        raise ValueError(f"Invalid SQL: tables not in schema → {missing_tables}")

    # 2) nie mieszaj nazw tabel z aliasami
    # jeśli tabela ma alias, to nie wolno używać 'table.column' w treści zapytania
    for a, t in alias_map.items():
        # ma alias tylko jeśli alias != table
        if a.lower() != t.lower():
            if re.search(rf'\b{re.escape(t)}\s*\.', sql, flags=re.IGNORECASE):
                raise ValueError(f"Do not mix table name '{t}' with its alias '{a}'.")

    # 3) sprawdź kolumny aliasowane
    col_refs = extract_column_refs(sql)
    unknown_aliases = [a for a, _ in col_refs if a.lower() not in alias_map_lc]
    if unknown_aliases:
        raise ValueError(f"Invalid SQL: unknown table aliases in column refs → {sorted(set(unknown_aliases))}")

    missing_cols: list[str] = []
    for a, c in col_refs:
        t_lc = alias_map_lc[a.lower()]
        if c.lower() not in schema_lc[t_lc]:
            missing_cols.append(f"{a}.{c} (table '{t_lc}')")

    if missing_cols:
        raise ValueError(f"Invalid SQL: columns not in schema → {missing_cols}")

def safe_execute_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """
    Wykonuje bezpiecznie SELECT na SQLite:
    - blokuje niebezpieczne polecenia (INSERT, DELETE, itp.),
    - waliduje zapytanie (validate_sql),
    - uruchamia zapytanie i zwraca wynik jako DataFrame,
    - dodatkowo klasyfikuje obrazki (YOLO) na podstawie kolumny 'image_url' lub 'vehicle_id'
      i dopisuje kolumnę 'detected_type'.
    """
    if FORBIDDEN_SQL.search(sql) or not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    schema = schema_dict_from_sqlite(conn)
    validate_sql(sql, schema)

    df = pd.read_sql_query(sql, conn)

    # dynamiczna ścieżka do folderu images (w tym samym katalogu co skrypt)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGES_DIR = os.path.join(BASE_DIR, "images")

    # jeśli mamy URL obrazów w wynikach
    if "image_url" in df.columns:
        clf = VehicleClassifier(backend=backend, yolo_weights=yolo_w)
        detected = []
        for url in df["image_url"]:
            if not url:
                detected.append("no_image")
                continue
            try:
                # jeśli url wygląda jak lokalna ścieżka, dodaj folder bazowy
                if not url.startswith("http://") and not url.startswith("https://"):
                    full_path = os.path.join(IMAGES_DIR, os.path.basename(url))
                    img = Image.open(full_path).convert("RGB")
                else:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    else:
                        detected.append("bad_url")
                        continue

                preds = clf.predict(img)
                if preds:
                    best = max(preds, key=lambda p: p.conf)
                    detected.append(best.label)
                else:
                    detected.append("unknown")
            except Exception:
                detected.append("error")
        df["detected_type"] = detected

    # albo jeśli mamy vehicle_id – wtedy dociągamy url z bazy
    elif "vehicle_id" in df.columns:
        clf = VehicleClassifier(backend=backend, yolo_weights=yolo_w)
        cur = conn.cursor()
        cur.execute("SELECT vehicle_id, image_url FROM vehicle_images;")
        img_map = {vid: url for vid, url in cur.fetchall()}

        detected = []
        for vid in df["vehicle_id"]:
            path = img_map.get(vid)
            if not path:
                detected.append("no_image")
                continue
            try:
                full_path = os.path.join(IMAGES_DIR, os.path.basename(path))
                img = Image.open(full_path).convert("RGB")
                preds = clf.predict(img)
                if preds:
                    best = max(preds, key=lambda p: p.conf)
                    detected.append(best.label)
                else:
                    detected.append("unknown")
            except Exception:
                detected.append("error")
        df["detected_type"] = detected

    return df


# ---------------- UI ----------------
st.set_page_config(page_title="Vehicle DB AI Demo", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle DB AI Demo")
st.caption("Image vehicle classification + NL→SQL over SQLite (local LLM via Ollama)")

# Sidebar: DB path & model
with st.sidebar:
    st.header("Settings")
    db_path = st.text_input("SQLite DB path", value="database.db")
    model_name = st.text_input("Ollama model", value=OLLAMA_MODEL_NAME)
    if model_name != OLLAMA_MODEL_NAME:
        OLLAMA_MODEL_NAME = model_name
    st.divider()
    st.subheader("Classifier backend")
    backend = st.selectbox("Backend", ["yolo"], index=0,
                           help="'yolo' = YOLO model")
    yolo_w = st.text_input("YOLO weights (if 'yolo')", value="yolo11m.pt")


# Tabs
img_tab, sql_tab = st.tabs(["🖼️ Image classification", "🤖 NL → SQL Agent"])

with img_tab:
    st.subheader("Image classification")
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up:
        image = Image.open(up).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
        clf = VehicleClassifier(backend=backend, yolo_weights=yolo_w)
        if st.button("Classify image", type="primary"):
            preds = clf.predict(image)
            if not preds:
                st.info("No target vehicle classes found or classifier not configured.")
            else:
                st.success(f"Detected {len(preds)} vehicles of target classes")
                rows = [(p.label, round(p.conf, 3)) for p in preds]
                st.dataframe(pd.DataFrame(rows, columns=["label", "confidence"]))

with sql_tab:
    st.subheader("Ask your database in natural language")

    if "schema_text" not in st.session_state or not st.session_state.schema_text:
        try:
            conn = sqlite3.connect(db_path)
            with conn:
                st.session_state.schema_text = compact_schema_from_sqlite(conn)
            conn.close()
        except Exception as e:
            st.session_state.schema_text = f"-- Failed to load schema: {e}"

    if "generated_sql" not in st.session_state:
        st.session_state.generated_sql = ""
    if "sql_editor" not in st.session_state:
        st.session_state.sql_editor = ""
    if "last_df" not in st.session_state:
        st.session_state.last_df = None

    question = st.text_area("Your question", value="Show all vehicles purchased by Anna Nowak.")

    if st.button("Generate SQL", type="primary"):
        try:
            conn = sqlite3.connect(db_path)
            with conn:
                schema_text = compact_schema_from_sqlite(conn)
            st.session_state.schema_text = schema_text
            with conn:
                sql = generate_and_test_sql(question, schema_text, conn , 3)
            st.session_state.generated_sql = sql
            st.session_state.sql_editor = sql
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.schema_text:
        st.markdown("**Database schema:**")
        st.code(st.session_state.schema_text, language="sql")

    if st.session_state.generated_sql:
        st.markdown("**Generated SQL:**")
        st.code(st.session_state.generated_sql, language="sql")

        st.session_state.sql_editor = st.text_area(
            "SQL to run",
            value=st.session_state.sql_editor,
            height=180
        )

        if st.button("Run SQL"):
            try:
                conn = sqlite3.connect(db_path)
                df = safe_execute_sql(conn, st.session_state.sql_editor)
                conn.close()
                st.session_state.last_df = df
            except Exception as e:
                st.error(f"Query failed: {e}")

    if st.session_state.last_df is not None:
        df = st.session_state.last_df
        st.success(f"Rows: {len(df)}")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="query_results.csv", mime="text/csv")


st.divider()
st.caption("Tip: Only SELECT queries are allowed for safety reasons.")
