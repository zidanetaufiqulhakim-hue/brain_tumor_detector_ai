from fastapi import FastAPI, UploadFile, File # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from PIL import Image
import io
import tensorflow as tf
import numpy as np
from utils.predict_img import predict_img
import base64

app = FastAPI()

# Load model once at startup
MODEL_PATH = "model/xception_brain_tumor_classifier_v2.h5"

# Load model with custom object scope to handle batch_shape compatibility
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
except (ValueError, TypeError) as e:
    if "batch_shape" in str(e):
        # Fallback: Use the .h5 model file if .keras fails
        print(f"Keras format loading failed: {e}. Attempting to load .h5 format...")
        model = tf.keras.models.load_model("model/xception_brain_tumor_classifier_v2.h5", compile=False)
    else:
        raise

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Brain Tumor Prediction"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """ Endpoint to predict brain tumor from an image. """
    try:
        # 1) Load image from request
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # 2) Predict using the utility function (includes grad-cam computation)
        prediction_result = predict_img(image, model=model, target_size=(224, 224))
        
        # 3) Convert grad-cam heatmap to base64 for JSON response
        heatmap = prediction_result["gradcam_heatmap"]
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_base64 = heatmap_uint8.tolist()
        
        # 4) Return response
        return JSONResponse({
            "status": "success",
            "probability_distribution": {
                "healthy": prediction_result["healthy_proba"],
                "pituitary_tumor": prediction_result["pituitary_tumor_proba"],
                "glioma_tumor": prediction_result["glioma_tumor_proba"],
                "meningioma_tumor": prediction_result["meningioma_tumor_proba"]
            },
            "predicted_class": prediction_result["predicted_class"],
            "gradcam_heatmap": heatmap_base64
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )
    