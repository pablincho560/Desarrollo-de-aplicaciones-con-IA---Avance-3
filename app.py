"""Interfaz de chat (Streamlit) del agente: historial, tools usadas y fuentes separadas por origen."""
import json

import streamlit as st

from src import config
from src.agent import Agent, new_session_id
from src.rag import RAGPipeline

st.set_page_config(page_title="Tutor Académico — Agente RAG", page_icon="🎓")
st.title("🎓 Tutor Académico — Agente RAG")
st.caption("Consulta el material de estudio, calcula y, si hace falta, busca en sitios web confiables.")


@st.cache_resource(show_spinner="Preparando la base de conocimiento...")
def load_agent() -> Agent:
    pipe = RAGPipeline()
    pipe.ensure_index()
    return Agent(pipe)


try:
    agent = load_agent()
except Exception as e:
    st.error(f"No se pudo iniciar el asistente: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()

web_activa = bool(config.get_tavily_key())
with st.sidebar:
    st.header("Configuración")
    st.write(f"Modelo: `{config.GENERATION_MODEL}`")
    st.write(f"Embeddings: `{config.EMBEDDING_MODEL}`")
    st.write(f"Máx. iteraciones por pregunta: {agent.max_iterations}")
    st.write(f"Máx. fragmentos por búsqueda: {config.MAX_FRAGMENTOS}")
    st.write("Herramientas: " + ", ".join(f"`{n}`" for n in agent.tools))
    st.write("🌐 Búsqueda web: " + ("activa" if web_activa else "**desactivada** (falta `TAVILY_API_KEY`)"))
    with st.expander("Sitios web permitidos"):
        st.write("\n".join(f"- {d}" for d in config.ALLOWED_WEB_DOMAINS))
    if st.button("🗑️ Nueva conversación"):
        st.session_state.messages = []
        st.session_state.session_id = new_session_id()
        st.rerun()


def render_tools(tool_calls: list[dict], uso: dict) -> None:
    titulo = f"🛠️ Herramientas usadas ({len(tool_calls)})" if tool_calls else "🛠️ Sin herramientas (respuesta directa)"
    with st.expander(titulo):
        for c in tool_calls:
            st.markdown(f"**{c['nombre']}** · {'✅' if c['ok'] else '⚠️'} · {c['duracion_s']} s")
            st.code(json.dumps(c["args"], ensure_ascii=False), language="json")
            st.caption(c["resumen"])
        st.caption(
            f"Llamadas al LLM: {uso['llm_calls']} · tokens: {uso['input_tokens']} entrada / "
            f"{uso['output_tokens']} salida · {uso['tiempo_s']} s"
            + (" · ⚠️ límite de iteraciones alcanzado" if uso.get("limite_alcanzado") else "")
        )


def render_sources(base: list[dict], web: list[dict]) -> None:
    if base:
        with st.expander(f"📚 Base de conocimientos ({len(base)})"):
            for f in base:
                st.markdown(f"**{f['documento']} — pág. {f['pagina']}** (similitud {f['similitud']})")
                st.caption(f["texto"])
    if web:
        with st.expander(f"🌐 Web ({len(web)})"):
            for w in web:
                st.markdown(f"**[{w['titulo']}]({w['url']})**  \n`web` · {w['url']}")
                st.caption(w["extracto"])


def render_extras(m: dict) -> None:
    render_tools(m.get("tool_calls", []), m["uso"])
    render_sources(m.get("fuentes_base", []), m.get("fuentes_web", []))


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_extras(m)

if prompt := st.chat_input("Escribe tu pregunta..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Pensando..."):
                result = agent.run(prompt, history, st.session_state.session_id)
        except Exception as e:
            st.error(f"Ocurrió un error al consultar el modelo: {e}")
            st.session_state.messages.pop()  # no dejamos la pregunta huérfana en el historial
            st.stop()
        st.markdown(result["respuesta"])
        msg = {
            "role": "assistant",
            "content": result["respuesta"],
            "tool_calls": result["tool_calls"],
            "fuentes_base": result["fuentes_base"],
            "fuentes_web": result["fuentes_web"],
            "uso": result["uso"],
        }
        render_extras(msg)
    st.session_state.messages.append(msg)
