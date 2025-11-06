# D:\aplicaciones red\mansory\mansory\pyservice\predict_service.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import uvicorn, json, numpy as np
from PIL import Image
import io
from tensorflow import keras
import numpy as np

APP_ROOT  = Path(__file__).resolve().parent
MODEL_DIR = Path(r"D:\aplicaciones red\mansory\mansory\storage\modelos")  # <— AQUÍ

BIN_MODEL_PATH   = MODEL_DIR / "bin_sin_con.keras"
MULTI_MODEL_PATH = MODEL_DIR / "multi_grietas_9.keras"
CLS_ALL_JSON     = MODEL_DIR / "classes_all.json"
CLS_MULTI_JSON   = MODEL_DIR / "classes_multi.json"
THRESHOLDS_JSON  = MODEL_DIR / "thresholds.json"

app = FastAPI(title="MansoryPredict", version="1.0.0")

bin_model   = keras.models.load_model(str(BIN_MODEL_PATH))
multi_model = keras.models.load_model(str(MULTI_MODEL_PATH))
class_names_all   = json.loads(CLS_ALL_JSON.read_text(encoding="utf-8"))["class_names"]
class_names_multi = json.loads(CLS_MULTI_JSON.read_text(encoding="utf-8"))["class_names"]
BIN_THRESHOLD = 0.5
if THRESHOLDS_JSON.exists():
    BIN_THRESHOLD = float(json.loads(THRESHOLDS_JSON.read_text(encoding="utf-8")).get("bin_threshold", 0.5))

def _prep(model, pil):
    H, W = model.input_shape[1], model.input_shape[2]
    x = np.array(pil.convert("RGB").resize((W, H))).astype("float32")/255.0
    return np.expand_dims(x, 0)

def _split(name):
    parts = name.split("_")
    return "_".join(parts[:-1]), parts[-1] if len(parts) > 1 else "—"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await image.read()))
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Imagen inválida: {e}"})

    p_con = float(bin_model.predict(_prep(bin_model, pil), verbose=0)[0][0])
    if p_con < BIN_THRESHOLD:
        return {"ok": True, "has_crack": False, "tipo": "sin_grietas", "severidad": "—", "conf_tipo": 1.0 - p_con}

    probs = multi_model.predict(_prep(multi_model, pil), verbose=0)[0].tolist()
    idx = int(np.argmax(probs))
    tipo, sev = _split(class_names_multi[idx])
    order = np.argsort(probs)[::-1][:3]
    top3 = [{"class": class_names_multi[i], "p": float(probs[i])} for i in order]

    return {"ok": True, "has_crack": True, "tipo": tipo, "severidad": sev, "conf_tipo": float(probs[idx]), "top3": top3}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)

