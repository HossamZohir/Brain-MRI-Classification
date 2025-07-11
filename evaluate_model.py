import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
HISTORY_PATH = "training_history.pth"  # added for history loading
# ==============

# ============================
# 📊 Plot Accuracy & Loss Curves
# ============================
def plot_training_curves(history):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ============================
# 🧪 Evaluate Model
# ============================
def evaluate_model(model, test_loader, class_names, history=None):
    model.eval()
    model.to(DEVICE)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\n📄 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ROC Curve & AUC
    print("\n📈 ROC Curves & AUC:")
    plt.figure(figsize=(10, 8))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curve & Average Precision Score
    print("\n📉 Precision-Recall Curves & AP Score:")
    plt.figure(figsize=(10, 8))
    precision, recall, ap_score = {}, {}, {}

    for i in range(len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        ap_score[i] = average_precision_score(all_labels == i, all_probs[:, i])
        plt.plot(recall[i], precision[i], lw=2,
                 label=f"{class_names[i]} (AP = {ap_score[i]:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (One-vs-Rest)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 🔁 Plot training curves if history is passed
    if history:
        print("\n📊 Plotting training curves...")
        plot_training_curves(history)

# Optional standalone run
if __name__ == "__main__":
    from data_loader import get_data_loaders
    from model_builder import build_model
    import os

    train_loader, val_loader, test_loader, class_names = get_data_loaders()
    model = build_model()

    # Load model weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"⚠️ Model file '{MODEL_PATH}' not found. Please train the model first.")
        exit(1)

    # Load training history if available
    if os.path.exists(HISTORY_PATH):
        history = torch.load(HISTORY_PATH)
        print(f"ℹ️ Loaded training history from '{HISTORY_PATH}'.")
    else:
        print(f"⚠️ Training history file '{HISTORY_PATH}' not found. Skipping training curves plot.")
        history = None

    evaluate_model(model, test_loader, class_names, history)
