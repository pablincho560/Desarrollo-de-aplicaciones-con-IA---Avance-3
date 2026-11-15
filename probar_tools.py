"""Prueba de las tools SIN LLM (captura para el informe).

    python probar_tools.py

buscar_en_base solo usa embeddings (no genera texto con el LLM). Las tareas no usan red.
"""
import json

from src.rag import RAGPipeline
from src.tools import build_tools

pipe = RAGPipeline()
pipe.ensure_index()
tools = {t.name: t for t in build_tools(pipe)}


def show(nombre: str, args: dict) -> None:
    print(f"\n>>> {nombre}.invoke({json.dumps(args, ensure_ascii=False)})")
    print(json.dumps(tools[nombre].invoke(args), ensure_ascii=False, indent=2))


show("buscar_en_base", {"consulta": "¿Qué es una pila y qué principio sigue?"})
show("calcular_operaciones", {"algoritmo": "bubble_sort", "n": 10000, "caso": "peor"})
show("generar_referencia_apa", {"documento": "guia_01_pilas.pdf"})
show("buscar_en_base", {"consulta": "x" * 1000})  # el limite de consulta se aplica en codigo
