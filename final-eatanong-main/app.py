import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv

# Assuming your model and class info setup remains the same
# Load the class names and descriptions from the CSV file
class_info = {}
with open('food-des.csv', 'r', encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        class_info[i] = (row[0], row[1])  # Assuming the first column is class names and the second is descriptions

# Initialize the model with the pre-trained weights
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Modify the fully connected layer
num_classes = 30  # Set this to the number of classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Now load your trained state_dict
state_dict = torch.load('model_state_dict.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Function to make prediction
def make_prediction(image_bytes):
    # Load and preprocess the image
    image = Image.open(image_bytes).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        max_prob, predicted = torch.max(probabilities, 0)

    # Retrieve the class name and description using the predicted index
    class_idx = predicted.item()
    class_name, description = class_info[class_idx]
    confidence = round(max_prob.item(), 2)  # Round the confidence to 2 decimal places

    return class_name, description, confidence

# Streamlit UI setup with designs
st.image("logo.png", use_column_width=True)
st.title("Filipino Food Image Classifier")

# Set background color with custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #F0F0F0;  /* Set your desired background color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display gif
# st.image("main-gif.gif", use_column_width=True)  # Replace "your_background.gif" with the actual file path or URL

# Additional content
st.markdown(
    "Welcome to Eatanong Heartwise, your essential companion for exploring Filipino cuisine! This innovative app helps determine if the food you're about to enjoy is generally heart-healthy or heart-detrimental. It combines advanced image recognition technology with a curated collection of hard-coded descriptions, enabling you to effortlessly predict and discover Filipino food classes."
)
st.markdown("Eatanong Heartwise is designed to assist in making heart-conscious dietary choices. With this app, users are empowered to make informed decisions about their diet while balancing their love for Filipino cuisine.")

# File uploader with design
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="fileuploader")

if uploaded_file is not None:
    # Display the uploaded image with design
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Make prediction
    prediction, description, confidence = make_prediction(uploaded_file)

    # Display the results with design
    st.subheader("Prediction Results:")
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence}%")
    st.write(f"**Description:** {description}")