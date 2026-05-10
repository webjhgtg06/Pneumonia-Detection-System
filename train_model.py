import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10

if __name__ == "__main__":
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        "dataset/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    test_data = test_gen.flow_from_directory(
        "dataset/test",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    # ── Model 1: EfficientNetB0 ──────────────────────────────────────────────
    base_model1 = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Phase 1 – train head only
    base_model1.trainable = False
    x1 = GlobalAveragePooling2D()(base_model1.output)
    vit_out = Dense(1, activation="sigmoid")(x1)
    vit_model = Model(base_model1.input, vit_out)
    vit_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    print("Training EfficientNetB0 (head only)...")
    vit_model.fit(train_data, epochs=EPOCHS // 2, validation_data=test_data)

    # Phase 2 – fine-tune full model
    base_model1.trainable = True
    vit_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    print("Fine-tuning EfficientNetB0...")
    vit_model.fit(train_data, epochs=EPOCHS // 2, validation_data=test_data)

    # Save model for GradCAM
    vit_model.save("efficientnet_model.h5")
    print("✅ EfficientNetB0 saved as efficientnet_model.h5")

    # ── Model 2: ResNet50 ────────────────────────────────────────────────────
    base_model2 = tf.keras.applications.ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Phase 1 – train head only
    base_model2.trainable = False
    x2 = GlobalAveragePooling2D()(base_model2.output)
    swin_out = Dense(1, activation="sigmoid")(x2)
    swin_model = Model(base_model2.input, swin_out)
    swin_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    print("Training ResNet50 (head only)...")
    swin_model.fit(train_data, epochs=EPOCHS // 2, validation_data=test_data)

    # Phase 2 – fine-tune full model
    base_model2.trainable = True
    swin_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    print("Fine-tuning ResNet50...")
    swin_model.fit(train_data, epochs=EPOCHS // 2, validation_data=test_data)

    # Save model for GradCAM
    swin_model.save("resnet_model.h5")
    print("✅ ResNet50 saved as resnet_model.h5")

    # ── Ensemble Evaluation ──────────────────────────────────────────────────
    y_true = test_data.classes
    vit_preds = vit_model.predict(test_data).flatten()
    swin_preds = swin_model.predict(test_data).flatten()
    ensemble_preds = (vit_preds + swin_preds) / 2
    final_preds = (ensemble_preds > 0.5).astype(int)

    print("\nFINAL ENSEMBLE RESULTS")
    print("Accuracy :", accuracy_score(y_true, final_preds))
    print("Precision:", precision_score(y_true, final_preds))
    print("Recall   :", recall_score(y_true, final_preds))
    print("F1 Score :", f1_score(y_true, final_preds))
