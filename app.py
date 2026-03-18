from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib, json
from routes.analyze import router as analyze_router

# Configuration
PATHS = {
    "stage1": "./models/cbc_stage1_model.joblib",
    "stage2": "./models/cbc_stage2_model.joblib",
    "encoder": "./models/cbc_stage2_label_encoder.joblib",
    "columns": "./models/cbc_feature_columns.joblib",
    "medians": "./models/cbc_feature_medians.joblib",
    "ontology": "./data/medical_ontology.json"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all at once
    try:
        app.state.artifacts = {
            "stage1_model": joblib.load(PATHS["stage1"]),
            "stage2_model": joblib.load(PATHS["stage2"]),
            "label_encoder": joblib.load(PATHS["encoder"]),
            "feature_columns": joblib.load(PATHS["columns"]),
            "feature_medians": joblib.load(PATHS["medians"]),
            "stage1_threshold": 0.6
        }
        with open(PATHS["ontology"], "r", encoding="utf-8") as f:
            app.state.artifacts["medical_ontology"] = json.load(f)
        print("✅ Backend ready: Models and Ontology loaded.")
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise RuntimeError(e)
    yield
    app.state.artifacts.clear()

app = FastAPI(title="CBC Decision Support", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)