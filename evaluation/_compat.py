"""Parche de compatibilidad: ragas 0.3.x importa clases de Vertex AI que langchain-community 0.4 ya no trae."""
import sys
import types


def _aplicar() -> None:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except Exception:
        mod = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # marcador vacio
            pass

        mod.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = mod
    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:
        try:
            import importlib

            llms = importlib.import_module("langchain_community.llms")
            llms.VertexAI = type("VertexAI", (), {})  # marcador vacio
        except Exception:
            pass


_aplicar()
