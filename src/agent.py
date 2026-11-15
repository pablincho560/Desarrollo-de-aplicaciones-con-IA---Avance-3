"""Agente con herramientas (Avance 3): bucle manual de tool calling sobre Gemini.

Flujo por pregunta:
    pregunta + historial -> LLM -> (¿pide tools?) -> el PROGRAMA ejecuta las tools -> resultado al LLM -> ... -> respuesta

Garantias aplicadas en codigo (no en el prompt):
  * maximo de iteraciones (llamadas al LLM) por pregunta; en la ultima se desactivan las tools para forzar respuesta;
  * politica "primero la base": buscar_en_web se rechaza si aun no se llamo a buscar_en_base en esta pregunta;
  * limites de cada tool (fragmentos, longitud, dominios web) en src/tools.py;
  * registro de cada pregunta en logs/tool_usage.jsonl.
"""
from __future__ import annotations

import time
import uuid

from google.genai import types

from . import config, llm, prompts, tool_log
from .tools import Tool, build_tools

FALLBACK = "No pude generar una respuesta. Intenta reformular la pregunta."


def _partes(resp) -> list:
    cand = (getattr(resp, "candidates", None) or [None])[0]
    content = getattr(cand, "content", None)
    return list(getattr(content, "parts", None) or [])


def _texto(resp) -> str:
    return "".join(p.text for p in _partes(resp) if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()


def _resumen(nombre: str, out: dict) -> str:
    if "error" in out:
        return f"error: {out['error']}"
    if nombre in ("buscar_en_base", "buscar_en_web"):
        n = len(out.get("resultados", []))
        return f"{n} {'fragmentos' if nombre == 'buscar_en_base' else 'resultados'}"
    if nombre == "calcular_operaciones":
        return f"{out.get('formula')} = {out.get('valor_texto')}"
    if nombre == "generar_referencia_apa":
        return out.get("referencia", "")
    return "ok"


class Agent:
    def __init__(self, pipeline, max_iterations: int | None = None):
        self.pipeline = pipeline
        self.tools: dict[str, Tool] = {t.name: t for t in build_tools(pipeline)}
        self.max_iterations = max_iterations or config.MAX_ITERATIONS
        self._tool_decl = types.Tool(function_declarations=[t.declaration() for t in self.tools.values()])

    # -- configuracion de cada llamada ------------------------------------------------------------
    def _gen_config(self, forzar_texto: bool):
        modo = types.FunctionCallingConfigMode.NONE if forzar_texto else types.FunctionCallingConfigMode.AUTO
        return types.GenerateContentConfig(
            system_instruction=prompts.AGENT_SYSTEM_PROMPT,
            temperature=0.2,
            tools=[self._tool_decl],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # el bucle es nuestro
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=modo)),
        )

    @staticmethod
    def _historial(history: list[dict] | None) -> list:
        out = []
        for m in (history or [])[-config.HISTORY_TURNS :]:
            if m.get("content"):
                role = "user" if m["role"] == "user" else "model"
                out.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        return out

    # -- bucle principal ----------------------------------------------------------------------------
    def run(self, question: str, history: list[dict] | None = None, session_id: str | None = None) -> dict:
        t0 = time.perf_counter()
        contents = self._historial(history) + [types.Content(role="user", parts=[types.Part(text=question)])]
        llamadas: list[dict] = []
        fuentes_base: list[dict] = []
        fuentes_web: list[dict] = []
        contextos: list[str] = []
        base_consultada = False
        tin = tout = n_llm = 0
        texto = ""
        iteracion = 0

        for iteracion in range(1, self.max_iterations + 1):
            ultima = iteracion == self.max_iterations
            resp = llm.generate_content(contents, self._gen_config(forzar_texto=ultima))  # LIMITE de iteraciones
            n_llm += 1
            u = llm.usage_of(resp)
            tin, tout = tin + u["input_tokens"], tout + u["output_tokens"]

            pedidas = [p.function_call for p in _partes(resp) if getattr(p, "function_call", None)]
            if not pedidas or ultima:
                texto = _texto(resp)
                break

            contents.append(resp.candidates[0].content)  # conserva el turno del modelo tal cual (incl. firmas de pensamiento)
            respuestas = []
            for fc in pedidas:
                nombre, args = fc.name, dict(fc.args or {})
                inicio = time.perf_counter()
                if nombre not in self.tools:
                    out = {"error": f"Herramienta desconocida: {nombre}"}
                elif nombre == "buscar_en_web" and not base_consultada:
                    out = {"error": "Política «primero la base»: llama antes a buscar_en_base y usa la web solo si la base no responde."}
                else:
                    out = self.tools[nombre].invoke(args)
                ok = "error" not in out
                if nombre == "buscar_en_base" and ok:
                    base_consultada = True
                    for r in out["resultados"]:
                        if not any(f["documento"] == r["documento"] and f["texto"] == r["texto"] for f in fuentes_base):
                            fuentes_base.append({k: r[k] for k in ("documento", "pagina", "texto", "similitud")})
                        contextos.append(r["texto"])
                elif nombre == "buscar_en_web" and ok:
                    for r in out["resultados"]:
                        fuentes_web.append({"titulo": r["titulo"], "url": r["url"], "extracto": r["extracto"]})
                        contextos.append(r["extracto"])
                llamadas.append(
                    {
                        "nombre": nombre,
                        "tipo": self.tools[nombre].kind if nombre in self.tools else "desconocida",
                        "args": args,
                        "ok": ok,
                        "resumen": _resumen(nombre, out),
                        "duracion_s": round(time.perf_counter() - inicio, 2),
                    }
                )
                respuestas.append(types.Part.from_function_response(name=nombre, response={"result": out}))
            contents.append(types.Content(role="user", parts=respuestas))

        uso = {
            "llm_calls": n_llm,
            "input_tokens": tin,
            "output_tokens": tout,
            "tiempo_s": round(time.perf_counter() - t0, 2),
            "iteraciones": iteracion,
            "limite_alcanzado": iteracion == self.max_iterations and bool(llamadas),
        }
        resultado = {
            "respuesta": texto or FALLBACK,
            "tool_calls": llamadas,
            "fuentes_base": fuentes_base,
            "fuentes_web": fuentes_web,
            "contextos": contextos,
            "uso": uso,
        }
        try:
            tool_log.log_interaction(
                {
                    "session_id": session_id,
                    "pregunta": question,
                    "modelo": config.GENERATION_MODEL,
                    "tools": [{k: c[k] for k in ("nombre", "args", "ok", "resumen", "duracion_s")} for c in llamadas],
                    "llamadas_llm": n_llm,
                    "tokens_entrada": tin,
                    "tokens_salida": tout,
                    "iteraciones": iteracion,
                    "limite_alcanzado": uso["limite_alcanzado"],
                    "tiempo_s": uso["tiempo_s"],
                    "respuesta": resultado["respuesta"][:300],
                }
            )
        except Exception:  # el registro nunca debe romper la respuesta
            pass
        return resultado


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


if __name__ == "__main__":
    from .rag import RAGPipeline

    pipe = RAGPipeline()
    pipe.ensure_index()
    agent = Agent(pipe)
    hist: list[dict] = []
    sid = new_session_id()
    print("Agente tutor (escribe 'salir' para terminar)")
    while True:
        q = input("\nTú: ").strip()
        if q.lower() in {"salir", "exit"}:
            break
        r = agent.run(q, hist, sid)
        print(f"\nTutor: {r['respuesta']}")
        for c in r["tool_calls"]:
            print(f"  🛠️ {c['nombre']}({c['args']}) -> {c['resumen']}")
        print(f"  [{r['uso']['llm_calls']} llamadas LLM, {r['uso']['tiempo_s']} s]")
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": r["respuesta"]}]
