"""Drug-name normalisation and regimen classification.

Real EHR medication fields are messy (brand names, strengths, free text). Normalising
on drug *names* rather than codes mirrors what is needed on real RWD.
"""
from __future__ import annotations

import re

# canonical agent -> regex over the free-text medication description
DRUG_PATTERNS: list[tuple[str, str]] = [
    ("carboplatin", r"carboplatin|paraplatin"),
    ("cisplatin", r"cisplatin|platinol"),
    ("pemetrexed", r"pemetrexed|alimta"),
    ("paclitaxel", r"paclitaxel|taxol"),
    ("docetaxel", r"docetaxel|taxotere"),
    ("gemcitabine", r"gemcitabine|gemzar"),
    ("pembrolizumab", r"pembrolizumab|keytruda"),
    ("nivolumab", r"nivolumab|opdivo"),
    ("atezolizumab", r"atezolizumab|tecentriq"),
    ("osimertinib", r"osimertinib|tagrisso"),
    ("erlotinib", r"erlotinib|tarceva"),
    ("ramucirumab", r"ramucirumab|cyramza"),
]
_COMPILED = [(name, re.compile(rx, re.I)) for name, rx in DRUG_PATTERNS]

# Free-text variants the generator writes into medications.csv
DESCRIPTION_VARIANTS: dict[str, list[str]] = {
    "carboplatin": ["Carboplatin 10 MG/ML Injectable Solution", "carboplatin", "Paraplatin"],
    "cisplatin": ["Cisplatin 1 MG/ML Injectable Solution", "CISPLATIN", "Platinol"],
    "pemetrexed": ["Pemetrexed 500 MG Injection", "pemetrexed disodium", "ALIMTA"],
    "paclitaxel": ["Paclitaxel 6 MG/ML Injectable Solution", "paclitaxel", "Taxol"],
    "docetaxel": ["Docetaxel 20 MG/ML Injectable Solution", "docetaxel", "Taxotere"],
    "gemcitabine": ["Gemcitabine 38 MG/ML Injectable Solution", "gemcitabine", "Gemzar"],
    "pembrolizumab": ["Pembrolizumab 25 MG/ML Injectable Solution", "pembrolizumab", "KEYTRUDA"],
    "nivolumab": ["Nivolumab 10 MG/ML Injectable Solution", "nivolumab", "Opdivo"],
    "atezolizumab": ["Atezolizumab 60 MG/ML Injectable Solution", "atezolizumab", "Tecentriq"],
    "osimertinib": ["Osimertinib 80 MG Oral Tablet", "osimertinib", "TAGRISSO"],
    "erlotinib": ["Erlotinib 150 MG Oral Tablet", "erlotinib", "Tarceva"],
    "ramucirumab": ["Ramucirumab 10 MG/ML Injectable Solution", "ramucirumab", "Cyramza"],
}

SUPPORTIVE_OR_OTHER = [
    "Dexamethasone 4 MG Oral Tablet", "Ondansetron 8 MG Oral Tablet",
    "Pegfilgrastim 6 MG/0.6ML Prefilled Syringe", "Metformin 500 MG Oral Tablet",
    "Lisinopril 10 MG Oral Tablet", "Atorvastatin 20 MG Oral Tablet",
]

# Agents whose pharmacology makes them interchangeable *within* a regimen (cis <-> carbo swap
# for toxicity is not a new line of therapy).
CANONICAL_FOR_LOT = {"carboplatin": "platinum", "cisplatin": "platinum"}
IO_AGENTS = {"pembrolizumab", "nivolumab", "atezolizumab"}
EGFR_TKI = {"osimertinib", "erlotinib"}


def normalise_drug(description: str) -> str | None:
    """Map free text to a canonical antineoplastic agent, or None if not an NSCLC antineoplastic."""
    if not isinstance(description, str):
        return None
    for name, rx in _COMPILED:
        if rx.search(description):
            return name
    return None


def lot_agent(drug: str) -> str:
    return CANONICAL_FOR_LOT.get(drug, drug)


def regimen_class(regimen: str) -> str:
    """Classify a '+'-joined regimen label into an analysis group."""
    agents = set(regimen.split("+"))
    if agents & EGFR_TKI:
        return "EGFR TKI"
    has_io = bool(agents & IO_AGENTS)
    has_plat = "platinum" in agents
    if has_io and has_plat:
        return "IO + chemo"
    if has_io:
        return "IO mono"
    if has_plat:
        return "Platinum chemo"
    return "Other"
