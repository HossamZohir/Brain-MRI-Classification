import torch
import torch.nn as nn
import timm

# === CONFIG START ===
NUM_CLASSES = 4
PRETRAINED = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === CONFIG END ===

def build_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    # Load pretrained ViT-B/16 from timm
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)

    # Replace the classifier head for our number of classes
    model.head = nn.Linear(model.head.in_features, num_classes)

    return model.to(DEVICE)
