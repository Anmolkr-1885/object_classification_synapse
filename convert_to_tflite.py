import tensorflow as tf
from tensorflow import keras
import os

MODEL_DIR = "models_cnn_igzo"
model_path = os.path.join(MODEL_DIR, "cnn_backbone.h5")

# Load model
model = keras.models.load_model(model_path)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (optional optimization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save
with open(os.path.join(MODEL_DIR, "model.tflite"), "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved!")