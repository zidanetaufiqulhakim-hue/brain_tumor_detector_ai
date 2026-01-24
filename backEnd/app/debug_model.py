import sys
import threading

# Limit to 1 thread at Python level
threading.stack_size(134217728)  # 128 MB

import os

# Set threading BEFORE tensorflow import  
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("✓ Environment variables set")

# Try importing with try/catch for better diagnostics
try:
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"✗ Error importing TensorFlow: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    MODEL_PATH = "model/xception_brain_tumor_classifier_v2.keras"
    print(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
