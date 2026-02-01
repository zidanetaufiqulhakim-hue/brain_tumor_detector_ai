from fastapi import FastAPI, UploadFile, File # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from PIL import Image
import io
import tensorflow as tf
import numpy as np
from utils.predict_img import predict_img
import base64
import os

# Disable threading issues on macOS
os.environ['TF_CPP_THREAD_MODE'] = 'serial'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configure TensorFlow for safer execution
# Configure TensorFlow for safer execution
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = FastAPI()

# Enable CORS - MUST be added first before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

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

@app.options("/predict")
async def options_predict():
    """Handle CORS preflight requests"""
    return {"message": "OK"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """ Endpoint to predict brain tumor from an image. """
    try:
        # 1) Load image from request
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # 2) Predict using the utility function (includes grad-cam computation)
        prediction_result = predict_img(image, model=model, target_size=(224, 224))
        
        # 3) Return response
        return JSONResponse({
            "status": "success",
            "probability_distribution": {
                "healthy": prediction_result["healthy_proba"],
                "pituitary_tumor": prediction_result["pituitary_tumor_proba"],
                "glioma_tumor": prediction_result["glioma_tumor_proba"],
                "meningioma_tumor": prediction_result["meningioma_tumor_proba"]
            },
            "predicted_class": prediction_result["predicted_class"],
            "gradcam_image": prediction_result["gradcam_image"] if prediction_result["gradcam_image"] else None
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )
    