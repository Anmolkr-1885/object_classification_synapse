import os
import io
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

# ── App Config ─────────────────────────────────
app = Flask(__name__, template_folder='templates')

# ── Paths ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_cnn_igzo")
TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")
RESULTS_PATH = os.path.join(MODEL_DIR, "igzo_cnn_results.pkl")

# ── Classes ───────────────────────────────────
CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

CLASS_EMOJI = {
    'airplane':'✈','automobile':'🚗','bird':'🐦','cat':'🐱','deer':'🦌',
    'dog':'🐶','frog':'🐸','horse':'🐴','ship':'🚢','truck':'🚛'
}

# CIFAR normalization
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# ── Globals ───────────────────────────────────
interpreter = None
W_dense = b_dense = W_out = b_out = None
igzo_results = {}

# ── Load Models (Safe) ─────────────────────────
def load_models():
    global interpreter, W_dense, b_dense, W_out, b_out, igzo_results

    print("🔄 Loading models...")

    try:
        # Load IGZO weights
        if os.path.exists(RESULTS_PATH):
            igzo_results = joblib.load(RESULTS_PATH)
            W_dense = igzo_results['W_dense']
            b_dense = igzo_results['b_dense']
            W_out   = igzo_results['W_out']
            b_out   = igzo_results['b_out']
            print("✅ IGZO weights loaded")
        else:
            print("❌ igzo_cnn_results.pkl not found")

        # Load TFLite model
        if os.path.exists(TFLITE_PATH):
            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            print("✅ TFLite model loaded")
        else:
            print("❌ model.tflite not found")

    except Exception as e:
        print("❌ Model loading failed:", e)

# Call safely
load_models()

# ── Preprocess ────────────────────────────────
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CIFAR_MEAN) / CIFAR_STD
    return arr[np.newaxis, ...]

# ── Feature Extraction ─────────────────────────
def extract_features(x):
    if interpreter is None:
        return None

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], x.astype('float32'))
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])

# ── IGZO Forward ──────────────────────────────
def igzo_forward(F, W1, b1, W2, b2):
    Z1 = F @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    Z2 = Z2 - Z2.max(axis=1, keepdims=True)
    E  = np.exp(Z2)
    return E / E.sum(axis=1, keepdims=True)

# ── Prediction ────────────────────────────────
def predict_image(img):
    if interpreter is None or W_dense is None:
        return None, None

    x = preprocess_image(img)
    features = extract_features(x)

    if features is None:
        return None, None

    probs = igzo_forward(features, W_dense, b_dense, W_out, b_out)[0]

    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]

    results = sorted([
        {
            'class': CLASS_NAMES[i],
            'prob': float(probs[i]) * 100,
            'emoji': CLASS_EMOJI[CLASS_NAMES[i]]
        }
        for i in range(10)
    ], key=lambda x: x['prob'], reverse=True)

    return pred_name, results

# ── Routes ────────────────────────────────────

# 🔥 Health check (IMPORTANT)
@app.route('/health')
def health():
    return "OK"

# 🔥 Home route
@app.route('/')
def index():
    print("🔥 HOME ROUTE HIT")
    model_ready = (interpreter is not None and W_dense is not None)
    acc = igzo_results.get('acc_final', 0)

    return render_template(
        'index.html',
        model_ready=model_ready,
        accuracy=f"{acc*100:.2f}" if acc else "N/A"
    )

# 🔥 Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))

        pred_name, results = predict_image(img)

        if pred_name is None:
            return jsonify({'error': 'Model not loaded'}), 500

        return jsonify({
            'prediction': pred_name,
            'emoji': CLASS_EMOJI[pred_name],
            'confidence': results[0]['prob'],
            'results': results
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)}), 500

# 🔥 Model info
@app.route('/model_info')
def model_info():
    return jsonify({
        'model_loaded': interpreter is not None,
        'accuracy': igzo_results.get('acc_final', 0) * 100
    })

# ── Run (Local only) ──────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)