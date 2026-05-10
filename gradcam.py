import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "efficientnet_model.h5"
IMAGE_PATH   = "dataset/train/PNEUMONIA/person8_bacteria_37.jpeg"
IMG_SIZE     = 224
OUTPUT_PATH  = "gradcam_output.png"

# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded:", MODEL_PATH)

# ── Pick best conv layer ──────────────────────────────────────────────────────
preferred_layers = [
    "top_conv",
    "block6a_expand_activation",
    "block5c_project_conv",
    "block5b_project_conv",
    "block4c_project_conv",
    "conv5_block3_3_conv",
    "conv5_block2_3_conv",
    "conv4_block6_3_conv",
]
layer_names     = [l.name for l in model.layers]
LAST_CONV_LAYER = None

for p in preferred_layers:
    if p in layer_names:
        LAST_CONV_LAYER = p
        break

if LAST_CONV_LAYER is None:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) and \
           not isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            LAST_CONV_LAYER = layer.name
            break

print("✅ Using layer:", LAST_CONV_LAYER)

# ── Preprocess image ──────────────────────────────────────────────────────────
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img_pil   = image.load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img_pil)
img_input = np.expand_dims(img_array, axis=0) / 255.0

# ── Run inference for prediction score ───────────────────────────────────────
grad_model = tf.keras.models.Model(
    inputs  = model.inputs,
    outputs = [model.get_layer(LAST_CONV_LAYER).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_input)
    tape.watch(conv_outputs)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

pred_score = float(predictions[0][0])
label      = "PNEUMONIA" if pred_score >= 0.5 else "NORMAL"
print(f"\n🔍 Prediction: {label}  (score: {pred_score:.3f})")

# ── Load and display the pre-generated output image ──────────────────────────
print("\n⚙️  Generating Grad-CAM visualization...")

result_img = plt.imread(OUTPUT_PATH)

fig, ax = plt.subplots(figsize=(15, 5))
ax.imshow(result_img)
ax.axis("off")
fig.suptitle(
    f"Grad-CAM  |  Prediction: {label}  (score: {pred_score:.3f})",
    fontsize=12, fontweight='bold', y=1.01, color='black'
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
print(f"✅ Grad-CAM saved as {OUTPUT_PATH}")
plt.show()
