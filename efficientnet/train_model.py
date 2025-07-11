import tensorflow as tf
from data_loader import train_generator, val_generator, class_names
from model_builder import build_model

# Define class weights (adjust based on class imbalance)
class_weight = {
    0: 1.0,  # glioma
    1: 1.0,  # healthy
    2: 1.2,  # meningioma
    3: 1.0   # pituitary
}

# Build model with option to unfreeze layers
model = build_model(unfreeze_layers=150, learning_rate=1e-3)

# 📦 Callbacks (model checkpoint, reduce LR, early stopping)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
]

# 🔁 Phase 1: Train head
print("🔁 Phase 1: Training with frozen base...")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    class_weight=class_weight,
    callbacks=callbacks
)

# 🔁 Phase 2: Fine-tune last 150 layers
print("🔁 Phase 2: Fine-tuning last 150 layers...")

base_model = model.layers[0]
fine_tune_at = len(base_model.layers) - 150
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 🧠 Train the fine-tuned model with early stopping
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weight,
    callbacks=callbacks
)
