from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Dataset directories
train_dir = "F:/hossam/work papers/Classification/dataset/train"
val_dir = "F:/hossam/work papers/Classification/dataset/val"
test_dir = "F:/hossam/work papers/Classification/dataset/test"

# Image size and batch size
IMG_SIZE = (300, 300)
BATCH_SIZE = 32

# Define ImageDataGenerator for training with advanced augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # EfficientNetV2 preprocessing
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# For validation and test, just preprocessing (no augmentation)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# Save class names for reporting etc.
class_names = list(train_generator.class_indices.keys())

print("✅ Generators created with advanced augmentation for training.")
