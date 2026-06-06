"""
Shared helper utilities.
"""

import re
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Medical terms to highlight in the UI
MEDICAL_TERMS = [
    # Clinical trial concepts
    "randomized", "placebo", "double-blind", "inclusion criteria", "exclusion criteria",
    "primary endpoint", "secondary endpoint", "adverse event", "serious adverse event",
    "informed consent", "protocol deviation", "clinical trial", "phase I", "phase II",
    "phase III", "phase IV", "investigational", "efficacy", "safety",
    # Pharmacology
    "dosage", "dose", "pharmacokinetics", "pharmacodynamics", "bioavailability",
    "half-life", "contraindication", "interaction", "metabolism", "excretion",
    "mechanism of action", "therapeutic", "toxicity", "overdose",
    # Disease areas
    "diabetes", "hypertension", "cardiovascular", "oncology", "cancer", "tumor",
    "inflammatory", "autoimmune", "infection", "sepsis", "pneumonia",
    # Statistics
    "p-value", "confidence interval", "hazard ratio", "odds ratio", "relative risk",
    "absolute risk", "number needed to treat", "intent-to-treat", "per-protocol",
    # Drug names (common examples)
    "metformin", "insulin", "aspirin", "warfarin", "statins", "atorvastatin",
    "lisinopril", "metoprolol", "amlodipine", "omeprazole",
]


def highlight_medical_terms(text: str, terms: List[str] = None) -> str:
    """
    Return text with medical terms wrapped in **markdown bold**.
    Used in Streamlit to draw attention to key medical vocabulary.
    """
    terms_to_highlight = terms or MEDICAL_TERMS
    for term in terms_to_highlight:
        pattern = re.compile(r"\b(" + re.escape(term) + r")\b", re.IGNORECASE)
        text = pattern.sub(r"**\1**", text)
    return text


def truncate_text(text: str, max_chars: int = 300, ellipsis: str = "...") -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + ellipsis


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 ** 2:.1f} MB"


def parse_publication_year(text: str) -> Optional[int]:
    """Extract a 4-digit year from free text."""
    matches = re.findall(r"\b(?:19|20)\d{2}\b", text)
    return int(matches[0]) if matches else None


def save_json(data: Any, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


DOCUMENT_TYPE_OPTIONS = [
    "Clinical Trial Report",
    "Clinical Study Protocol",
    "Medical Guideline",
    "Drug Information Document",
    "Research Paper",
    "FDA Guidance Document",
    "ICH-GCP Guideline",
    "Patient Information Leaflet",
    "Pharmacoeconomic Research Paper",
    "Other",
]

DISEASE_AREA_OPTIONS = [
    "General",
    "Cardiovascular",
    "Diabetes / Metabolic",
    "Oncology",
    "Neurology",
    "Respiratory",
    "Infectious Disease",
    "Autoimmune / Rheumatology",
    "Gastroenterology",
    "Endocrinology",
    "Dermatology",
    "Ophthalmology",
    "Psychiatry",
    "Rare Disease",
    "Other",
]

EXAMPLE_QUESTIONS = [
    "What are the inclusion criteria in this clinical trial?",
    "What adverse effects are reported for Metformin?",
    "Summarize the treatment guidelines for hypertension.",
    "What endpoints were used in the diabetes study?",
    "What is the recommended dosage of the drug?",
    "What are the contraindications mentioned in this document?",
    "Describe the study design and methodology.",
    "What were the primary efficacy results?",
    "What safety monitoring procedures were in place?",
    "What are the pharmacokinetic properties described?",
]
