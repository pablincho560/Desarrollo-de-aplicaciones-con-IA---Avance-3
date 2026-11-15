"""Cliente de Gemini: embeddings, generacion y contabilidad de uso (llamadas y tokens)."""
import time

from google import genai
from google.genai import types

from . import config

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = config.get_api_key()
        _client = genai.Client(
            vertexai=api_key.startswith("AQ."),
            api_key=api_key,
        )
    return _client


def _with_retry(fn, retries: int = 5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # backoff por limites de cuota


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_size: int = 50) -> list[list[float]]:
    """Devuelve un embedding por texto. task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY."""
    client = get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _with_retry(
            lambda b=batch: client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=b,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        )
        vectors.extend([e.values for e in resp.embeddings])
        time.sleep(0.5)
    return vectors


def generate_content(contents, gen_config):
    """Una llamada al LLM con reintentos. Devuelve la respuesta cruda (la usa el bucle del agente)."""
    client = get_client()
    return _with_retry(
        lambda: client.models.generate_content(model=config.GENERATION_MODEL, contents=contents, config=gen_config)
    )


def usage_of(resp) -> dict:
    """Uso de UNA llamada: {'llm_calls', 'input_tokens', 'output_tokens'} (tokens de razonamiento incluidos en output)."""
    meta = getattr(resp, "usage_metadata", None)
    tin = (getattr(meta, "prompt_token_count", 0) or 0) if meta else 0
    tout = ((getattr(meta, "candidates_token_count", 0) or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)) if meta else 0
    return {"llm_calls": 1, "input_tokens": tin, "output_tokens": tout}


def generate_with_usage(prompt: str, system_instruction: str | None = None, temperature: float = 0.2) -> tuple[str, dict]:
    resp = generate_content(
        prompt,
        types.GenerateContentConfig(system_instruction=system_instruction, temperature=temperature),
    )
    return (resp.text or "").strip(), usage_of(resp)


def generate(prompt: str, system_instruction: str | None = None, temperature: float = 0.2) -> str:
    return generate_with_usage(prompt, system_instruction, temperature)[0]
