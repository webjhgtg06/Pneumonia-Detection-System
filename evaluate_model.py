import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
IMG_SIZE = 224
BATCH_SIZE = 16
MODEL_PATH = "model.h5"   
TEST_DIR = "dataset/test"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded from", MODEL_PATH)
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)
y_true = test_data.classes
y_prob = model.predict(test_data).ravel()
y_pred = (y_prob > 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
precision, recall, _ = precision_recall_curve(y_true, y_prob)
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.show()

print("✅ Graphs saved:")
print(" - confusion_matrix.png")
print(" - roc_curve.png")
print(" - precision_recall_curve.png")
