"""Tools del agente (Avance 3).

Todas las tools:
  - devuelven un dict serializable (nunca lanzan excepciones al LLM: devuelven {"error": ...});
  - aplican sus LIMITES en codigo (no en el prompt);
  - se pueden probar sin LLM con  tool.invoke({...}).

Tools:
  buscar_en_base          (busqueda) fragmentos de la base de conocimientos, con documento y pagina
  calcular_operaciones    (tarea)    numero teorico de operaciones segun la complejidad de un algoritmo
  generar_referencia_apa  (tarea)    referencia APA de un documento de la base
  buscar_en_web           (web)      buscador (Tavily) restringido a sitios permitidos
"""
from __future__ import annotations

import inspect
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from . import config

SUPPORTED = {".pdf", ".docx", ".txt", ".html", ".htm"}


# ---------------------------------------------------------------------------
# Infraestructura
# ---------------------------------------------------------------------------
@dataclass
class Param:
    name: str
    type: str  # "STRING" | "INTEGER"
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class Tool:
    name: str
    kind: str  # "busqueda" | "tarea" | "web"
    params: list[Param]
    func: Callable[..., dict]

    @property
    def description(self) -> str:
        """El docstring de la funcion ES la descripcion que ve el LLM."""
        return inspect.cleandoc(self.func.__doc__ or "")

    def invoke(self, args: dict | None = None, **kwargs: Any) -> dict:
        """Ejecuta la tool sin LLM. Nunca lanza: los errores vuelven como {'error': ...}."""
        data = {**(args or {}), **kwargs}
        try:
            return self.func(**data)
        except TypeError as e:
            return {"error": f"Argumentos inválidos para {self.name}: {e}"}
        except Exception as e:  # noqa: BLE001 - la tool debe degradar con elegancia
            return {"error": f"{type(e).__name__}: {e}"}

    def declaration(self):
        """Declaracion de funcion para Gemini (tool calling)."""
        from google.genai import types

        props = {
            p.name: types.Schema(type=types.Type[p.type], description=p.description, enum=p.enum)
            for p in self.params
        }
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties=props,
                required=[p.name for p in self.params if p.required],
            ),
        )


def _cut(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


# ---------------------------------------------------------------------------
# 1) Busqueda en la base de conocimientos
# ---------------------------------------------------------------------------
def make_buscar_en_base(pipeline) -> Tool:
    def buscar_en_base(consulta: str) -> dict:
        """Busca en la base de conocimientos (guías de estudio de estructuras de datos y algoritmos:
        pilas, colas, listas enlazadas, árboles BST, grafos y ordenamiento) y devuelve los fragmentos
        más relevantes, cada uno con su documento y página para poder citarlo.
        Úsala SIEMPRE que la pregunta sea de contenido académico del curso, ANTES de responder y antes
        de considerar la búsqueda web. Si la pregunta tiene varios temas, llámala una vez por tema.
        No la uses para saludos ni charla casual. La consulta debe ser una frase autónoma y corta."""
        consulta = (consulta or "").strip()
        if not consulta:
            return {"error": "La consulta está vacía."}
        consulta = consulta[: config.MAX_QUERY_CHARS]
        fragmentos = pipeline.retrieve(consulta)[: config.MAX_FRAGMENTOS]  # LIMITE 1: nº de fragmentos
        return {
            "origen": "base",
            "resultados": [
                {
                    "fuente": f"{f['documento']}, pág. {f['pagina']}",
                    "documento": f["documento"],
                    "pagina": f["pagina"],
                    "similitud": f["similitud"],
                    "texto": _cut(f["texto"], config.MAX_CHARS_FRAGMENTO),  # LIMITE 2: largo del fragmento
                }
                for f in fragmentos
            ],
        }

    return Tool(
        "buscar_en_base",
        "busqueda",
        [Param("consulta", "STRING", "Frase autónoma y corta que describe lo que se busca en el material.")],
        buscar_en_base,
    )


# ---------------------------------------------------------------------------
# 2) Tarea del dominio A: calculo de operaciones segun la complejidad
# ---------------------------------------------------------------------------
def _cuad(n: int) -> int:
    return n * (n - 1) // 2


# algoritmo -> caso -> (formula legible, funcion). Las formulas corresponden a las complejidades
# de las guias (Bubble O(n^2), Merge O(n log n), Quick O(n log n) prom. / O(n^2) peor,
# BST O(log n) prom. / O(n) peor, lista enlazada O(n)).
_MODELOS: dict[str, dict[str, tuple[str, Callable[[int], float]]]] = {
    "bubble_sort": {
        "peor": ("n(n-1)/2 comparaciones — O(n²)", _cuad),
        "promedio": ("n(n-1)/2 comparaciones — O(n²)", _cuad),
    },
    "merge_sort": {
        "peor": ("n·log2(n) comparaciones aprox. — O(n log n)", lambda n: n * math.log2(n)),
        "promedio": ("n·log2(n) comparaciones aprox. — O(n log n)", lambda n: n * math.log2(n)),
    },
    "quick_sort": {
        "peor": ("n(n-1)/2 comparaciones — O(n²)", _cuad),
        "promedio": ("n·log2(n) comparaciones aprox. — O(n log n)", lambda n: n * math.log2(n)),
    },
    "bst_buscar": {
        "peor": ("n pasos — O(n)", lambda n: n),
        "promedio": ("log2(n) pasos — O(log n)", lambda n: math.log2(n)),
    },
    "lista_buscar": {
        "peor": ("n nodos visitados — O(n)", lambda n: n),
        "promedio": ("n/2 nodos visitados — O(n)", lambda n: n / 2),
    },
}
_ALIAS = {
    "bubble": "bubble_sort", "burbuja": "bubble_sort", "bubblesort": "bubble_sort",
    "ordenamientoburbuja": "bubble_sort",
    "merge": "merge_sort", "mergesort": "merge_sort", "mezcla": "merge_sort",
    "quick": "quick_sort", "quicksort": "quick_sort", "rapido": "quick_sort",
    "bst": "bst_buscar", "arbol": "bst_buscar", "bstbuscar": "bst_buscar", "arbolbinariodebusqueda": "bst_buscar",
    "lista": "lista_buscar", "listaenlazada": "lista_buscar", "listabuscar": "lista_buscar",
}
_N_MAX = 10**12


def calcular_operaciones(algoritmo: str, n: int, caso: str = "peor") -> dict:
    """Calcula el número teórico de operaciones (comparaciones o pasos) de un algoritmo o estructura
    para un tamaño de entrada n, según su complejidad. Úsala cuando el usuario pida una cifra concreta
    ("¿cuántas comparaciones hace bubble sort con 10 000 elementos?", "¿cuántos pasos toma buscar en un
    BST de un millón de nodos?"). NUNCA calcules estos números de memoria: llama a esta tool.
    Extrae algoritmo, n y caso de la pregunta. Es una estimación teórica, no un tiempo medido."""
    clave = _norm(algoritmo)
    alg = clave if clave in _MODELOS else _ALIAS.get(clave)
    if alg is None:
        return {"error": f"Algoritmo no soportado: '{algoritmo}'. Opciones: {', '.join(_MODELOS)}."}
    caso = _norm(caso or "peor")
    caso = {"peorcaso": "peor", "worst": "peor", "average": "promedio", "promedio": "promedio", "peor": "peor"}.get(caso, caso)
    if caso not in _MODELOS[alg]:
        return {"error": f"Caso no soportado: '{caso}'. Usa 'peor' o 'promedio'."}
    try:
        n = int(float(n))
    except (TypeError, ValueError):
        return {"error": f"n debe ser un entero, recibí: {n!r}"}
    if not 1 <= n <= _N_MAX:
        return {"error": f"n debe estar entre 1 y {_N_MAX}."}
    formula, fn = _MODELOS[alg][caso]
    valor = fn(n)
    valor = int(valor) if float(valor).is_integer() else round(float(valor), 2)
    return {
        "origen": "calculo",
        "algoritmo": alg,
        "caso": caso,
        "n": n,
        "formula": formula,
        "valor": valor,
        "valor_texto": f"{valor:,}",
        "nota": "Estimación teórica según la complejidad; no es un tiempo medido.",
    }


def make_calcular_operaciones() -> Tool:
    return Tool(
        "calcular_operaciones",
        "tarea",
        [
            Param("algoritmo", "STRING", "Algoritmo o estructura a analizar.", enum=list(_MODELOS)),
            Param("n", "INTEGER", "Tamaño de la entrada (número de elementos o nodos), extraído de la pregunta."),
            Param("caso", "STRING", "Caso a calcular. Por defecto 'peor'.", required=False, enum=["peor", "promedio"]),
        ],
        calcular_operaciones,
    )


# ---------------------------------------------------------------------------
# 3) Tarea del dominio B: referencia APA de un documento de la base
# ---------------------------------------------------------------------------
_AUTORES_BASURA = {"", "anonymous", "unknown", "none", "desconocido", "anonimo"}


def _archivos_base() -> list[Path]:
    return sorted(f for f in config.DATA_DIR.iterdir() if f.suffix.lower() in SUPPORTED)


def _titulo_documento(path: Path) -> tuple[str, str]:
    """Titulo (primera linea no vacia del documento) y autor de los metadatos, si los hay."""
    autor = ""
    texto = ""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        texto = (reader.pages[0].extract_text() or "") if reader.pages else ""
        meta = reader.metadata
        autor = (getattr(meta, "author", "") or "").strip("() ").strip() if meta else ""
    elif path.suffix.lower() == ".txt":
        texto = path.read_text(encoding="utf-8", errors="ignore")
    else:
        from .ingest import load_document  # docx / html

        paginas = load_document(path)
        texto = paginas[0]["texto"] if paginas else ""
    titulo = next((ln.strip() for ln in texto.splitlines() if ln.strip()), path.stem)
    return titulo, ("" if autor.lower() in _AUTORES_BASURA else autor)


def generar_referencia_apa(documento: str, autor: str = "", anio: str = "") -> dict:
    """Genera la referencia en formato APA (7.ª ed.) de un documento que está en la base de conocimientos,
    leyendo su título real del archivo. Úsala cuando el usuario pida "la referencia APA", "cómo citar"
    o "la bibliografía" de una guía o documento. NUNCA inventes la referencia: llama a esta tool.
    Argumentos: documento (nombre de archivo o parte reconocible del nombre, ej. 'guia_01_pilas.pdf' o
    'pilas'), autor en formato 'Apellido, N.' (solo si el usuario lo da) y año (solo si el usuario lo da).
    Si no se conocen autor o año, la referencia usa el título en la posición del autor y '(s. f.)'."""
    consulta = _norm(documento)
    archivos = _archivos_base()
    nombres = [f.name for f in archivos]
    if not consulta:
        return {"error": "Indica el documento.", "documentos_disponibles": nombres}
    exactos = [f for f in archivos if _norm(f.name) == consulta or _norm(f.stem) == consulta]
    parciales = exactos or [f for f in archivos if consulta in _norm(f.name)]
    if len(parciales) != 1:
        motivo = "No encontré ese documento" if not parciales else "Hay varios documentos que coinciden"
        return {"error": f"{motivo}: '{documento}'.", "documentos_disponibles": nombres}
    path = parciales[0]

    titulo, autor_meta = _titulo_documento(path)
    autor = (autor or "").strip() or autor_meta
    anio = (anio or "").strip() or "s. f."
    tipo = f"Guía de estudio en {path.suffix.lstrip('.').upper()}"
    if autor:
        ref = f"{autor} ({anio}). *{titulo}* [{tipo}]. {config.APA_FUENTE}."
    else:
        ref = f"*{titulo}* [{tipo}]. ({anio}). {config.APA_FUENTE}."
    nota = []
    if not autor:
        nota.append("El documento no declara autor: se usó el título en su lugar.")
    if anio == "s. f.":
        nota.append("Sin año de publicación: se usó (s. f.).")
    return {
        "origen": "tarea",
        "documento": path.name,
        "referencia": ref,
        "nota": " ".join(nota) or "Referencia completa.",
    }


def make_generar_referencia_apa() -> Tool:
    return Tool(
        "generar_referencia_apa",
        "tarea",
        [
            Param("documento", "STRING", "Nombre del archivo o parte reconocible del nombre (ej. 'guia_01_pilas.pdf')."),
            Param("autor", "STRING", "Autor en formato 'Apellido, N.'. Solo si el usuario lo indica.", required=False),
            Param("anio", "STRING", "Año de publicación. Solo si el usuario lo indica.", required=False),
        ],
        generar_referencia_apa,
    )


# ---------------------------------------------------------------------------
# 4) Busqueda web controlada (politica "primero la base")
# ---------------------------------------------------------------------------
def _dominio_permitido(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in config.ALLOWED_WEB_DOMAINS)


def make_buscar_en_web() -> Tool:
    def buscar_en_web(consulta: str) -> dict:
        """Busca en la web, SOLO en una lista fija de sitios oficiales o confiables, información pública
        o actual del dominio (por ejemplo, la versión más reciente de un lenguaje, documentación oficial
        o una definición general). Política "primero la base": úsala únicamente DESPUÉS de llamar a
        buscar_en_base y solo si la base NO contiene la respuesta. No la uses para preguntas que la base
        ya responde, ni para temas ajenos al dominio. La consulta debe ser corta (máx. 200 caracteres) y
        no debe incluir texto copiado de la base. Devuelve máx. 3 resultados con título, URL y extracto."""
        q = (consulta or "").strip()
        if not q:
            return {"error": "La consulta está vacía."}
        if len(q) > config.WEB_MAX_QUERY_CHARS:  # LIMITE: nada de pegar texto del corpus en el buscador
            return {"error": f"Consulta demasiado larga (máx. {config.WEB_MAX_QUERY_CHARS} caracteres). Acórtala."}
        key = config.get_tavily_key()
        if not key:
            return {"error": "Búsqueda web no disponible: falta TAVILY_API_KEY."}
        from tavily import TavilyClient

        resp = TavilyClient(api_key=key).search(
            query=q,
            max_results=config.WEB_MAX_RESULTADOS,
            include_domains=list(config.ALLOWED_WEB_DOMAINS),  # lista fijada en codigo
            search_depth="basic",
        )
        resultados = []
        for r in resp.get("results", []):
            url = r.get("url", "")
            if not _dominio_permitido(url):  # defensa en profundidad: filtramos tambien aqui
                continue
            resultados.append(
                {
                    "origen": "web",
                    "titulo": _cut(r.get("title", ""), 150),
                    "url": url,
                    "extracto": _cut(r.get("content", ""), config.WEB_MAX_CHARS_EXTRACTO),
                }
            )
            if len(resultados) >= config.WEB_MAX_RESULTADOS:
                break
        out: dict = {"origen": "web", "resultados": resultados}
        if not resultados:
            out["aviso"] = "Sin resultados en los sitios permitidos."
        return out

    return Tool(
        "buscar_en_web",
        "web",
        [Param("consulta", "STRING", "Consulta corta (máx. 200 caracteres) sobre información pública o actual.")],
        buscar_en_web,
    )


def build_tools(pipeline) -> list[Tool]:
    return [
        make_buscar_en_base(pipeline),
        make_calcular_operaciones(),
        make_generar_referencia_apa(),
        make_buscar_en_web(),
    ]
