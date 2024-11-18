# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import streamlit as st
# import matplotlib.pyplot as plt

# # Load the trained model from the models directory
# model_path = os.path.join("model", "skin_analysis_model_vgg16.h5")
# model = load_model(model_path)

# # Define the data generator for preprocessing (same as during training)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(rescale=1.0 / 255)


# # Function to predict skin condition from an uploaded image
# def predict_skin_condition(img_path):
#     # Load the image
#     img = Image.open(img_path)

#     # Convert the image to RGB (removes alpha channel if present)
#     img = img.convert("RGB")

#     # Resize image to match input size for the model
#     img = img.resize((128, 128))

#     # Convert image to numpy array
#     img_array = np.array(img)

#     # Reshape and normalize image for model input
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Rescale the pixel values to [0, 1]

#     # Make prediction
#     prediction = model.predict(img_array)

#     # Get the class with the highest probability
#     predicted_class = np.argmax(prediction, axis=1)[0]

#     # Map the predicted class to the actual label
#     class_labels = [
#         "acne",
#         "healthy",
#         "pigmented",
#         "textured",
#     ]  # Ensure this matches your label order
#     predicted_label = class_labels[predicted_class]

#     return predicted_label, prediction[0][predicted_class]


# # Streamlit UI for image upload
# st.title("Skin Condition Analysis")
# st.write("Upload an image of your skin to analyze its condition.")

# # Upload the image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     # Predict the skin condition
#     predicted_label, probability = predict_skin_condition(uploaded_file)

#     # Display the prediction result
#     st.write(f"Predicted Label: {predicted_label}")
#     st.write(f"Prediction Confidence: {probability * 100:.2f}%")

#     # Display the result with a plot
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     ax.axis("off")  # Hide axes
#     ax.set_title(f"Predicted: {predicted_label} ({probability * 100:.2f}%)")
#     st.pyplot(fig)
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model from the models directory
model_path = os.path.join("model", "skin_analysis_model_vgg16.h5")

# Option 1: Use a pre-trained VGG16 model
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
# x = Flatten()(base_model.output)
# x = Dense(512, activation='relu')(x)
# x = Dense(4, activation='softmax')(x)  # 4 classes
# model = Model(inputs=base_model.input, outputs=x)

# Option 2: Fine-tuning ResNet50 for better accuracy
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
x = Dense(4, activation="softmax")(x)  # 4 classes
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Define the data generator for preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)


# Function to predict skin condition from an uploaded image
def predict_skin_condition(img_path):
    # Load the image
    img = Image.open(img_path)
    img = img.convert("RGB")  # Convert RGBA images to RGB

    # Resize image to match input size for the model
    img = img.resize((128, 128))

    # Convert image to numpy array
    img_array = np.array(img)

    # Reshape and normalize image for model input
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale pixel values

    # Make prediction
    prediction = model.predict(img_array)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map the predicted class to the actual label
    class_labels = ["acne", "healthy", "pigmented", "textured"]
    predicted_label = class_labels[predicted_class]

    return predicted_label, prediction[0][predicted_class]


# Streamlit UI for image upload
st.title("Skin Condition Analysis")
st.write("Upload an image of your skin to analyze its condition.")

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict the skin condition
    predicted_label, probability = predict_skin_condition(uploaded_file)

    # Display the prediction result
    st.write(f"Predicted Label: {predicted_label}")
    st.write(f"Prediction Confidence: {probability * 100:.2f}%")

    # Display the result with a plot
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")  # Hide axes
    ax.set_title(f"Predicted: {predicted_label} ({probability * 100:.2f}%)")
    st.pyplot(fig)

    # Save the model using model checkpointing and early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        "model/skin_analysis_model_vgg16.h5", save_best_only=True, monitor="val_loss"
    )

    # Sample training with augmented data (replace `train_images` and `train_labels` with your actual data)
    # model.fit(datagen.flow(train_images, train_labels), epochs=20, validation_data=(val_images, val_labels), callbacks=[early_stopping, model_checkpoint])
