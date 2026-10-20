"""Observabilité minimale : chaque question est enregistrée dans un fichier JSONL.

On garde la latence, les tokens consommés, les outils appelés et les sources.
C'est le même principe que Langfuse, en version locale et simplifiée.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class Tracer:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None

    def log(self, event: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
