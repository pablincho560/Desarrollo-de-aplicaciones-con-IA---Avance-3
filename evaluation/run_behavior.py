"""Casos de comportamiento del agente (esperado escrito ANTES en behavior_cases.json).

Uso:  python -m evaluation.run_behavior
Salida: evaluation/results/comportamiento.json y comportamiento.md (tabla para el informe).
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from src import config, prompts
from src.agent import Agent
from src.ingest import SUPPORTED, load_document
from src.rag import RAGPipeline
from src.tools import _dominio_permitido

HERE = Path(__file__).parent
RES = HERE / "results"


def _compacta(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def hay_volcado(respuesta: str, ventana: int = 120, paso: int = 30) -> bool:
    """True si la respuesta reproduce un tramo literal de >=120 caracteres de algun documento del corpus."""
    r = _compacta(respuesta)
    for f in sorted(config.DATA_DIR.iterdir()):
        if f.suffix.lower() not in SUPPORTED:
            continue
        for pg in load_document(f):
            t = _compacta(pg["texto"])
            if any(t[i : i + ventana] in r for i in range(0, max(1, len(t) - ventana), paso)):
                return True
    return False


def evaluar(caso: dict, r: dict) -> list[str]:
    """Devuelve la lista de reglas incumplidas (vacia = cumple)."""
    reglas, resp = caso["reglas"], r["respuesta"]
    usadas = Counter(c["nombre"] for c in r["tool_calls"])
    fallos = []
    if "*" in reglas.get("tools_prohibidas", []) and usadas:
        fallos.append(f"llamó tools: {dict(usadas)}")
    for t in reglas.get("tools_prohibidas", []):
        if t != "*" and usadas.get(t):
            fallos.append(f"llamó {t} (prohibida)")
    for t in reglas.get("tools_requeridas", []):
        if not usadas.get(t):
            fallos.append(f"no llamó {t}")
    for pat in reglas.get("respuesta_contiene_todos", []):
        if not re.search(pat, resp, re.I):
            fallos.append(f"falta /{pat}/")
    alguno = reglas.get("respuesta_contiene_alguno")
    if alguno and not any(re.search(p, resp, re.I) for p in alguno):
        fallos.append(f"ninguno de {alguno}")
    if reglas.get("web_solo_sitios_permitidos") and not all(_dominio_permitido(w["url"]) for w in r["fuentes_web"]):
        fallos.append("URL fuera de sitios permitidos")
    if reglas.get("sin_volcado") and (hay_volcado(resp) or len(resp) > 1500):
        fallos.append("volcó contenido del corpus o respondió demasiado largo")
    if reglas.get("respuesta_contiene_mensaje_no_info") and prompts.NO_INFO_MESSAGE not in resp:
        fallos.append("no devolvió el mensaje exacto de 'sin información'")
    return fallos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(HERE / "behavior_cases.json"))
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args()

    casos = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    pipe = RAGPipeline()
    pipe.ensure_index()
    agent = Agent(pipe)

    salida = []
    for c in casos:
        print(f"[{c['id']}] {c['pregunta']}")
        r = agent.run(c["pregunta"], session_id="comportamiento")
        fallos = evaluar(c, r)
        salida.append(
            {
                "id": c["id"],
                "tipo": c["tipo"],
                "pregunta": c["pregunta"],
                "esperado": c["esperado"],
                "obtenido": r["respuesta"],
                "tools": [{"nombre": t["nombre"], "args": t["args"], "ok": t["ok"]} for t in r["tool_calls"]],
                "llamadas_llm": r["uso"]["llm_calls"],
                "cumple": not fallos,
                "fallos": fallos,
            }
        )
        print("   ->", "CUMPLE" if not fallos else f"NO CUMPLE: {fallos}")
        time.sleep(args.pause)

    RES.mkdir(exist_ok=True)
    (RES / "comportamiento.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    def celda(t: str, n: int = 160) -> str:
        t = re.sub(r"\s+", " ", t).replace("|", "/")
        return t if len(t) <= n else t[: n - 1] + "…"

    md = ["| Caso | Tipo | Esperado | Obtenido | Tools usadas | Llamadas LLM | ¿Cumple? |", "|---|---|---|---|---|---|---|"]
    for s in salida:
        tools = "; ".join(f"{t['nombre']}({json.dumps(t['args'], ensure_ascii=False)})" for t in s["tools"]) or "(ninguna)"
        md.append(
            f"| {s['id']} | {s['tipo']} | {celda(s['esperado'])} | {celda(s['obtenido'])} | {celda(tools, 200)} | "
            f"{s['llamadas_llm']} | {'✅' if s['cumple'] else '❌ ' + celda('; '.join(s['fallos']), 80)} |"
        )
    (RES / "comportamiento.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n{sum(s['cumple'] for s in salida)}/{len(salida)} casos cumplen. Tabla en {RES / 'comportamiento.md'}")


if __name__ == "__main__":
    main()
