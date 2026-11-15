"""Evaluacion Ragas del AGENTE sobre el mismo conjunto de preguntas del Avance 2.

Los retrieved_contexts son los fragmentos que devolvieron las tools (base y web).

Uso:  python -m evaluation.run_ragas_agent --tag agente
Luego: python -m evaluation.compare_results --before mejora --after agente
"""
import argparse
import json
import time
from pathlib import Path

from evaluation.run_ragas import HERE, evaluate_rows, save_results
from src import config
from src.agent import Agent
from src.rag import RAGPipeline

SIN_CONTEXTO = "(el agente no recuperó fragmentos para esta pregunta)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="agente")
    ap.add_argument("--questions", default=str(HERE / "eval_questions.json"))
    ap.add_argument("--max-iterations", type=int, default=config.MAX_ITERATIONS)
    ap.add_argument("--pause", type=float, default=1.0, help="segundos entre preguntas (cuotas de la API)")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    pipe = RAGPipeline()
    pipe.ensure_index()
    agent = Agent(pipe, args.max_iterations)

    rows, extras = [], []
    for q in questions:
        print(f"[{q['id']}/{len(questions)}] {q['question']}")
        r = agent.run(q["question"], session_id=f"ragas-{args.tag}")
        rows.append(
            {
                "user_input": q["question"],
                "retrieved_contexts": r["contextos"] or [SIN_CONTEXTO],  # Ragas no admite listas vacias
                "response": r["respuesta"],
                "reference": q["ground_truth"],
            }
        )
        extras.append(r)
        time.sleep(args.pause)

    df = evaluate_rows(rows)
    df.insert(0, "id", [q["id"] for q in questions])
    df["in_corpus"] = [q["in_corpus"] for q in questions]
    df["tools_usadas"] = [", ".join(c["nombre"] for c in r["tool_calls"]) or "(ninguna)" for r in extras]
    df["llamadas_llm"] = [r["uso"]["llm_calls"] for r in extras]
    df["tokens_entrada"] = [r["uso"]["input_tokens"] for r in extras]
    df["tokens_salida"] = [r["uso"]["output_tokens"] for r in extras]
    df["tiempo_s"] = [r["uso"]["tiempo_s"] for r in extras]

    n = len(extras)
    cfg = {
        "tipo": "agente",
        "max_iterations": args.max_iterations,
        "max_fragmentos": config.MAX_FRAGMENTOS,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
        "generation_model": config.GENERATION_MODEL,
    }
    costo = {
        "llamadas_llm_prom": round(sum(r["uso"]["llm_calls"] for r in extras) / n, 2),
        "tokens_prom": round(sum(r["uso"]["input_tokens"] + r["uso"]["output_tokens"] for r in extras) / n, 1),
        "tiempo_prom_s": round(sum(r["uso"]["tiempo_s"] for r in extras) / n, 2),
    }
    save_results(df, args.tag, cfg, extra={"costo": costo})


if __name__ == "__main__":
    main()
