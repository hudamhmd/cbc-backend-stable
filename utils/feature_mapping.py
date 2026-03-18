import pandas as pd
from typing import Dict, List, Any

MODEL_COLS = [
    "wbc", "rbc", "hemoglobin", "hematocrit", "mcv", 
    "mch", "mchc", "platelets", "lymp_pct", 
    "neut_pct", "lymp_abs", "neut_abs",
]

ALIASES = {
    "wbc": ["wbc", "white_blood_cells", "wbc_count"],
    "rbc": ["rbc", "red_blood_cells", "rbc_count"],
    "hemoglobin": ["hemoglobin", "hgb", "hb"],
    "hematocrit": ["hematocrit", "hct"],
    "mcv": ["mcv"], "mch": ["mch"], "mchc": ["mchc"],
    "platelets": ["platelets", "plt", "platelet_count"],
    "lymp_pct": ["lymp_pct", "lymph_pct", "lymphocytes_percent", "lymph%"],
    "neut_pct": ["neut_pct", "neutrophils_percent", "neut%"],
    "lymp_abs": ["lymp_abs", "lymph_abs", "alc"],
    "neut_abs": ["neut_abs", "neutrophils_abs", "anc"],
}

def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace("%", "pct")

ALIAS_TO_CANON = { _norm(n): canon for canon, names in ALIASES.items() for n in names }

def map_input_to_model_features(raw_values: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in (raw_values or {}).items():
        nk = _norm(str(k))
        canon = ALIAS_TO_CANON.get(nk) or ALIAS_TO_CANON.get(nk.replace("percent", "pct"))
        if canon:
            try: out[canon] = float(v)
            except: continue
    return out

def validate_numeric_ranges(values: Dict[str, float]) -> List[str]:
    warnings = []
    ranges = {
        "wbc": (0.1, 100), "rbc": (1.0, 8.0), "hemoglobin": (3.0, 22.0),
        "hematocrit": (10.0, 70.0), "mcv": (50.0, 130.0), "platelets": (5.0, 1000.0)
    }
    for key, (min_v, max_v) in ranges.items():
        if key in values and (values[key] < min_v or values[key] > max_v):
            warnings.append(f"{key.upper()} value {values[key]} is outside typical range ({min_v}-{max_v})")
    return warnings

def build_feature_vector(values_mapped: Dict[str, float], feature_columns: List[str], medians: Dict[str, float]) -> pd.DataFrame:
    row = {col: values_mapped.get(col, float(medians.get(col, 0.0))) for col in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)