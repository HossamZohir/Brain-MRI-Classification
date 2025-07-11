import torch
import torch.nn as nn
import timm

# === CONFIG START ===
NUM_CLASSES = 6
PRETRAINED = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === CONFIG END ===

def build_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    # Load pretrained ConvNeXt Tiny from timm
    model = timm.create_model('convnext_tiny', pretrained=pretrained)

    # Make sure we flatten the output before classifier
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),        # From (B, 768, 7, 7) → (B, 768, 1, 1)
        nn.Flatten(),                   # From (B, 768, 1, 1) → (B, 768)
        nn.Linear(model.num_features, num_classes)
    )

    return model.to(DEVICE)


#models = timm.list_models("convnext*")
#print(models)