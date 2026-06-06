"""
Persistent document registry backed by a JSON file.
Tracks ingested documents and their metadata independently of ChromaDB.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = "data/document_registry.json"


class DocumentManager:
    """Simple JSON-backed registry for ingested documents."""

    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Registry file corrupted, starting fresh.")
        return {}

    def _save(self):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)

    def register(self, document_id: str, metadata: dict, ingest_summary: dict):
        self._registry[document_id] = {
            "document_id": document_id,
            "registered_at": datetime.now().isoformat(),
            **metadata,
            "ingest_summary": ingest_summary,
        }
        self._save()
        logger.info(f"Registered document: {document_id}")

    def get(self, document_id: str) -> Optional[dict]:
        return self._registry.get(document_id)

    def remove(self, document_id: str) -> bool:
        if document_id in self._registry:
            del self._registry[document_id]
            self._save()
            return True
        return False

    def list_all(self) -> List[dict]:
        return list(self._registry.values())

    def count(self) -> int:
        return len(self._registry)

    def get_filter_options(self) -> dict:
        """Collect unique values for each filterable field."""
        options = {
            "document_type": set(),
            "disease_area": set(),
            "drug_name": set(),
            "publication_year": set(),
        }
        for doc in self._registry.values():
            for field in options:
                val = doc.get(field)
                if val and val not in ("N/A", "", "Unknown", "General"):
                    options[field].add(str(val))

        return {k: sorted(v) for k, v in options.items()}
