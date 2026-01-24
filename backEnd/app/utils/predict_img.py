import tensorflow as tf
import numpy as np
from PIL import Image

def compute_gradcam(img_array, model):
    """
    Grad-CAM for Sequential(Xception) model — SAFE version
    """
    # 1. Extract base Xception model
    base_model = model.get_layer("xception")

    # 2. Last convolutional layer INSIDE Xception
    last_conv_layer = base_model.get_layer("block14_sepconv2_act")

    # 3. Rebuild classifier head manually
    x = base_model.output
    x = model.layers[1](x)  # global_max_pooling2d
    x = model.layers[2](x)  # dense
    predictions = model.layers[3](x)  # dense_1

    # 4. Build Grad-CAM model (PURE Functional)
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, predictions]
    )

    # 5. Gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def predict_img(image: Image.Image, model, target_size=(224, 224)):

    # Preprocess image
    img = image.convert("RGB").resize(target_size)
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    # Predict
    y_proba = model.predict(img_array)

    #  classes=["healthy","pituitary","glioma","meningioma"],
    label_map = {
        0: "healthy",
        1: "pituitary_tumor",
        2: "glioma_tumor",
        3: "meningioma_tumor"
    }

    # label the prediction
    predicted_class = label_map[np.argmax(y_proba)]

    # ✅ Grad-CAM (tensor, not PIL)
    heatmap = compute_gradcam(img_array, model)

    return {
        "healthy_proba": round(float(y_proba[0][0]), 4),
        "pituitary_tumor_proba": round(float(y_proba[0][1]), 4),
        "glioma_tumor_proba": round(float(y_proba[0][2]), 4),
        "meningioma_tumor_proba": round(float(y_proba[0][3]), 4),
        "predicted_class": predicted_class,
        "gradcam_heatmap": heatmap if predicted_class != "healthy" else np.zeros((7,7)),  # Empty heatmap for healthy
    }


