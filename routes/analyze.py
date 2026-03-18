import numpy as np
from fastapi import APIRouter, HTTPException, Request
from schemas.predict_request import PredictRequest
from schemas.predict_response import PredictResponse
from utils.feature_mapping import map_input_to_model_features, build_feature_vector, validate_numeric_ranges, _norm

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    artifacts = getattr(request.app.state, "artifacts", None)
    if not artifacts:
        raise HTTPException(status_code=500, detail="Server artifacts not initialized")

    warnings = []
    values_mapped = map_input_to_model_features(req.cbc_values)
    warnings.extend(validate_numeric_ranges(values_mapped))
    
    # Check minimum fields
    required = ["hemoglobin", "wbc", "platelets", "mcv"]
    missing = [f for f in required if f not in values_mapped]
    if missing: warnings.append(f"Missing key fields: {', '.join(missing)}. Prediction might be unstable.")

    X = build_feature_vector(values_mapped, artifacts["feature_columns"], artifacts["feature_medians"])

    # --- Stage 1 ---
    stage1_model = artifacts["stage1_model"]
    threshold = artifacts.get("stage1_threshold", 0.6)
    proba = stage1_model.predict_proba(X)[0][1] if hasattr(stage1_model, "predict_proba") else float(stage1_model.predict(X)[0])
    
    cbc_related = proba >= threshold
    stage1_out = {"cbc_related_probability": round(float(proba), 3), "cbc_related": bool(cbc_related), "threshold": threshold}

    ontology = artifacts["medical_ontology"]

    # --- NON-CBC PATH ---
    if not cbc_related:
        rules = ontology.get("non_cbc_related", {})
        diagnosis_hint = " ".join([str(v).lower() for v in (req.context or {}).values() if isinstance(v, str)])
        matched = [r for r in rules.get("patterns", []) if any(k.lower() in diagnosis_hint for k in r.get("keywords", []))]
        if not matched and rules.get("default"): matched = [rules["default"]]

        return PredictResponse(
            stage1=stage1_out, path="NON_CBC", top_predictions=[], ontology_support=[],
            urgent_attention=any(r.get("red_flag", False) for r in matched),
            recommended_tests=[{"test": t, "reason": r.get("reason"), "priority": r.get("priority", 2)} for r in matched for t in r.get("recommended_tests", [])],
            specialty=list({r.get("specialty") for r in matched if r.get("specialty")}),
            red_flags=[r.get("red_flag_text", "") for r in matched if r.get("red_flag")],
            warnings=warnings, disclaimer=ontology.get("global_disclaimer", "Clinical support only.")
        )

    # --- CBC PATH (Stage 2) ---
    stage2_model = artifacts["stage2_model"]
    label_encoder = artifacts["label_encoder"]
    
    probs = stage2_model.predict_proba(X)[0]
    top_idx = np.argsort(probs)[::-1][:req.top_k]
    decoded_labels = label_encoder.inverse_transform([stage2_model.classes_[i] for i in top_idx])

    top_preds = []
    all_tests, all_specialties, all_red_flags = [], set(), []

    for i, label in enumerate(decoded_labels):
        condition_key = _norm(str(label))
        info = ontology.get("cbc_conditions", {}).get(condition_key, {})
        
        # Enrichment
        p_val = float(probs[top_idx[i]])
        top_preds.append({
            "rank": i+1, "condition": str(label), "probability": p_val,
            "probability_percent": round(p_val * 100, 2),
            "likely_causes": info.get("likely_causes", []),
            "confirmatory_tests": info.get("confirmatory_tests", [])
        })
        
        all_specialties.add(info.get("specialty")) if info.get("specialty") else None
        all_red_flags.extend(info.get("red_flags", []))
        for t in info.get("confirmatory_tests", []):
            all_tests.append(t if isinstance(t, dict) else {"test": t, "priority": 2, "reason": f"Evaluate {label}"})

    return PredictResponse(
        stage1=stage1_out, path="CBC", top_predictions=top_preds, ontology_support=[],
        urgent_attention=len(all_red_flags) > 0,
        recommended_tests={t['test']: t for t in all_tests}.values(), # Simple deduplicate
        specialty=list(all_specialties), red_flags=list(set(all_red_flags)),
        warnings=warnings, disclaimer=ontology.get("global_disclaimer", "Clinical support only.")
    )