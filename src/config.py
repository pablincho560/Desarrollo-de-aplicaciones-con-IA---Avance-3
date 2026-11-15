"""Configuracion central. Las claves se leen de variables de entorno o de st.secrets."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
HISTORY_TURNS = 6  # mensajes previos que se envian al LLM

# ---------------------------------------------------------------------------
# Avance 3: limites del agente. Viven en el CODIGO, no en el prompt.
# ---------------------------------------------------------------------------
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))  # llamadas al LLM por pregunta
MAX_FRAGMENTOS = 4  # maximo de fragmentos que devuelve buscar_en_base por llamada
MAX_CHARS_FRAGMENTO = 800  # longitud maxima de cada fragmento devuelto (= chunk_size por defecto)
MAX_QUERY_CHARS = 300  # longitud maxima de la consulta a la base

# Busqueda web (Tavily). La lista la fija el codigo; el LLM no puede cambiarla.
WEB_MAX_RESULTADOS = 3
WEB_MAX_CHARS_EXTRACTO = 300
WEB_MAX_QUERY_CHARS = 200  # una consulta corta no permite "colar" texto del corpus hacia el buscador
ALLOWED_WEB_DOMAINS = (
    "python.org",  # documentacion y versiones oficiales de Python
    "docs.oracle.com",  # documentacion oficial de Java (estructuras de datos, Collections)
    "cppreference.com",  # referencia de C/C++ y de la STL
    "developer.mozilla.org",  # referencia oficial de JavaScript
    "wikipedia.org",  # definiciones generales y de teoria de la computacion
    "geeksforgeeks.org",  # explicaciones de algoritmos y estructuras de datos
    "ocw.mit.edu",  # cursos abiertos del MIT
    "khanacademy.org",  # cursos introductorios de algoritmos
)

# Referencia APA
APA_FUENTE = os.getenv("APA_FUENTE", "Material de estudio del curso")

# Registro de uso de tools
LOG_PATH = Path(os.getenv("TOOL_LOG_PATH", str(BASE_DIR / "logs" / "tool_usage.jsonl")))


def get_secret(name: str, aliases: tuple[str, ...] = (), required: bool = True) -> str | None:
    """Busca una clave en variables de entorno y, si no esta, en st.secrets (Streamlit Cloud)."""
    names = (name, *aliases)
    for n in names:
        value = os.getenv(n)
        if value:
            return value
    try:
        import streamlit as st

        for n in names:
            value = st.secrets.get(n)
            if value:
                return value
    except Exception:
        pass
    if required:
        raise RuntimeError(
            f"Falta {name}. Defínela en .env, como variable de entorno "
            "o en los Secrets de la plataforma de despliegue."
        )
    return None


def get_api_key() -> str:
    return get_secret("GOOGLE_API_KEY", ("GEMINI_API_KEY",))


def get_tavily_key() -> str | None:
    """Clave del buscador web. Es opcional: sin ella la tool web se desactiva con un mensaje claro."""
    return get_secret("TAVILY_API_KEY", required=False)
