import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# === CONFIG ===
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# === Grad-CAM Implementation ===
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap)

        return heatmap

# === Utility Functions ===
def apply_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + image
    return np.uint8(superimposed_img)

def preprocess_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return input_tensor, np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))

def load_model(build_model_func):
    model = build_model_func()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def get_medical_explanation(predicted_class):
    explanations = {
        "glioma": (
            "🧠 Diagnosis: Glioma\n"
            "   - A diffuse infiltrative tumor usually in the cerebral hemispheres.\n"
            "   - Grad-CAM should highlight irregular masses, possibly causing a midline shift or edema."
        ),
        "meningioma": (
            "🧠 Diagnosis: Meningioma\n"
            "   - Typically an extra-axial tumor with well-defined borders.\n"
            "   - Grad-CAM often focuses on peripheral/dural regions where tumors are attached to the meninges."
        ),
        "pituitary": (
            "🧠 Diagnosis: Pituitary Tumor\n"
            "   - Located in the sella turcica, midline below the brain.\n"
            "   - Grad-CAM usually highlights the center base of the brain, where the pituitary gland lies."
        ),
        "healthy": (
            "🧠 Diagnosis: Healthy Brain\n"
            "   - No significant abnormalities.\n"
            "   - Grad-CAM should not show any concentrated attention on a particular abnormal region."
        )
    }
    return explanations.get(predicted_class.lower(), "No medical explanation available.")

# === Example Usage ===
if __name__ == "__main__":
    from model_builder import build_model

    # === SET IMAGE PATH ===
    image_path = "F:/hossam/work papers/Classification/dataset/test/pituitary/0010.jpg"  # CHANGE THIS

    # === CLASS LABELS ===
    class_labels = ["glioma", "healthy", "meningioma", "pituitary"]

    # === Load model and prepare Grad-CAM ===
    model = load_model(build_model)
    target_layer = model.stages[-1]
    grad_cam = GradCAM(model, target_layer)

    # === Preprocess image and run model ===
    input_tensor, raw_image = preprocess_image(image_path)
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1).squeeze().cpu().detach().numpy()
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_labels[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    # === Grad-CAM Visualization ===
    heatmap = grad_cam.generate(input_tensor, class_idx=predicted_class_idx)
    result = apply_heatmap(raw_image, heatmap)

    # === Show Grad-CAM result ===
    plt.imshow(result)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
    plt.show()

    # === Print Detailed XAI Explanation ===
    print("\n========== MODEL EXPLANATION ==========")
    print(f"🧠 Predicted Class: {predicted_class.upper()}")
    print(f"📊 Confidence: {confidence*100:.2f}%")
    print(f"🗺️ Grad-CAM highlights regions that influenced this decision the most.")
    print(f"💡 Interpretation:")
    print(f"    - The model focused on the regions in red/yellow.")
    print(f"    - These areas contain patterns consistent with '{predicted_class}' class.")
    print(f"    - Blue/dark regions were not significant for this decision.")
    print(f"\n🩺 Medical Insight:\n{get_medical_explanation(predicted_class)}")
