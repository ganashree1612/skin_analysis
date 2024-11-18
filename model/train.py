import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Get the current working directory and construct the data path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../data")  # Your data folder path
data_dir = os.path.abspath(data_dir)

print(f"Using data directory: {data_dir}")

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize images to [0, 1]
    validation_split=0.2,  # Use 20% of data for validation
    rotation_range=40,  # Random rotation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Zooming in/out
    horizontal_flip=True,  # Randomly flip images
    fill_mode="nearest",  # Fill pixels after transformations
)

# Training and validation generators
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),  # Resize images
    batch_size=16,  # Increased batch size
    class_mode="categorical",  # Multi-class classification
    subset="training",  # Only use training data
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=16,  # Increased batch size
    class_mode="categorical",
    subset="validation",  # Validation data
)

# Print number of images in each generator
print("Number of training samples:", train_gen.samples)
print("Number of validation samples:", val_gen.samples)

# Load VGG16 model + higher level layers
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(train_gen.num_classes, activation="softmax")(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Model summary
model.summary()

# Train the model
history = model.fit(train_gen, epochs=20, validation_data=val_gen)

# Save the model
model.save("skin_analysis_model_vgg16.h5")
