# 📦 Import necessary libraries
import tensorflow as tf
from data_loader import test_generator, class_names  # ⬅️ FIXED: Import test_generator instead
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Load the best saved model
model = tf.keras.models.load_model("best_model.h5")

# 📊 Predict using test_generator
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# 🧾 Classification report
print("🧾 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# 📌 Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
