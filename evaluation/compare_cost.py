"""Costo: pipeline fijo (Avance 2) vs. agente (Avance 3) sobre las MISMAS preguntas.

Mide, por pregunta: llamadas al LLM, tokens (entrada/salida) y tiempo de respuesta.
Uso:  python -m evaluation.compare_cost            # las 16 preguntas
      python -m evaluation.compare_cost --n 8      # solo las primeras 8
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.agent import Agent
from src.rag import RAGPipeline

HERE = Path(__file__).parent
RES = HERE / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args()

    questions = json.loads((HERE / "eval_questions.json").read_text(encoding="utf-8"))[: args.n]
    pipe = RAGPipeline()
    pipe.ensure_index()
    agent = Agent(pipe)

    filas = []
    for q in questions:
        print(f"[{q['id']}] {q['question']}")
        p = pipe.answer(q["question"])["uso"]
        a = agent.run(q["question"], session_id="costo")["uso"]
        filas.append(
            {
                "id": q["id"],
                "pipeline_llm": p["llm_calls"], "pipeline_tokens": p["input_tokens"] + p["output_tokens"], "pipeline_s": p["tiempo_s"],
                "agente_llm": a["llm_calls"], "agente_tokens": a["input_tokens"] + a["output_tokens"], "agente_s": a["tiempo_s"],
            }
        )
        time.sleep(args.pause)

    df = pd.DataFrame(filas)
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / "costo_detalle.csv", index=False)
    resumen = pd.DataFrame(
        {
            "Pipeline fijo (Avance 2)": [df.pipeline_llm.mean(), df.pipeline_tokens.mean(), df.pipeline_s.mean()],
            "Agente (Avance 3)": [df.agente_llm.mean(), df.agente_tokens.mean(), df.agente_s.mean()],
        },
        index=["Llamadas al LLM (prom.)", "Tokens por pregunta (prom.)", "Tiempo de respuesta, s (prom.)"],
    ).round(2)
    resumen["Factor (agente/pipeline)"] = (resumen["Agente (Avance 3)"] / resumen["Pipeline fijo (Avance 2)"]).round(2)
    (RES / "costo_resumen.json").write_text(resumen.to_json(force_ascii=False, indent=2), encoding="utf-8")
    print("\n" + (resumen.to_markdown() if hasattr(resumen, "to_markdown") else resumen.to_string()))


if __name__ == "__main__":
    main()
