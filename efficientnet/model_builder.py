import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B2

IMG_SHAPE = (300, 300, 3)
NUM_CLASSES = 4

def build_model(unfreeze_layers=150, learning_rate=1e-4):
    base_model = EfficientNetV2B2(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SHAPE
    )

    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the last `unfreeze_layers` layers for fine-tuning
    fine_tune_at = len(base_model.layers) - unfreeze_layers
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
