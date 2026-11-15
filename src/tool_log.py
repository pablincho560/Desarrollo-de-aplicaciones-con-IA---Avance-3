"""Registro de uso de tools: una linea JSON por pregunta en logs/tool_usage.jsonl.

Cada registro guarda: que tools llamo el agente, con que argumentos, cuantas llamadas hizo al LLM,
tokens y tiempo.   Ver los ultimos registros:   python -m src.tool_log --tail 5
"""
import argparse
import json
from datetime import datetime, timezone

from . import config


def log_interaction(record: dict) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def tail(n: int = 5) -> list[dict]:
    if not config.LOG_PATH.exists():
        return []
    lines = config.LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(ln) for ln in lines[-n:]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=5)
    for rec in tail(ap.parse_args().tail):
        print(json.dumps(rec, ensure_ascii=False, indent=2))
