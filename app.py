import streamlit as st
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd

CLASS_NAMES = ["ants", "bees"]
MODEL_NAME = "tlcv_model.pt"

# epoch log 
training_logs = [
    [0, 0.7823, 0.5943, 1.0876, 0.5752],
    [1, 0.6160, 0.7582, 0.1686, 0.9477],
    [2, 0.3391, 0.8730, 0.1862, 0.9477],
    [3, 0.4823, 0.7910, 0.6825, 0.7647],
    [4, 0.5514, 0.8197, 0.1860, 0.9412],
    [5, 0.3754, 0.8361, 0.2825, 0.9085],
    [6, 0.4072, 0.8074, 0.1806, 0.9477],
    [7, 0.3800, 0.8525, 0.2024, 0.9477],
    [8, 0.3990, 0.8361, 0.2384, 0.9150],
    [9, 0.2648, 0.8934, 0.2395, 0.9150],
    [10, 0.3342, 0.8525, 0.1971, 0.9412],
    [11, 0.3263, 0.8689, 0.1950, 0.9412],
    [12, 0.3444, 0.8525, 0.1959, 0.9346],
    [13, 0.3523, 0.8361, 0.2386, 0.9150],
    [14, 0.2886, 0.8689, 0.1878, 0.9346],
    [15, 0.2510, 0.8893, 0.1947, 0.9477],
    [16, 0.3271, 0.8525, 0.1991, 0.9477],
    [17, 0.3313, 0.8443, 0.1915, 0.9477],
    [18, 0.3579, 0.8443, 0.2148, 0.9281],
    [19, 0.3306, 0.8566, 0.2100, 0.9412],
]

def predict(model_name, img_path, threshold=0.8):
    model = torch.load(model_name, map_location="cpu", weights_only=False)

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    image = Image.open(img_path)
    preprocessed_image = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(preprocessed_image)
        probs = F.softmax(output, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)

    if max_prob.item() < threshold:
        return "Can't classify"
    else:
        return CLASS_NAMES[pred_idx]

def prediction_tab(model_name):
    st.header("Image Classification via Transfer Learning :: ResNet")
    uploaded_file = st.file_uploader(
        "Upload image of ants or bees (.jpeg/.jpg/.png only)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Predict"):
            predicted_class = predict(model_name, uploaded_file)
            st.subheader(f"Prediction: {predicted_class}")

def logs_tab():
    st.header("Notes")
    df = pd.DataFrame(training_logs, columns=["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])
    st.dataframe(df, use_container_width=True)
    best_val_acc = df["Val Acc"].max()
    st.markdown(f"**Best Validation Accuracy:** {best_val_acc:.4f}")

    st.subheader("Info")
    st.write({
        "Dataset": "ants & bees dataset (Hymenoptera_data)",
        "Number of Classes": len(CLASS_NAMES),
        "Classes": CLASS_NAMES,
        "Image Size": "224x224 (center cropped)",
        "Dataset Path": "./data/hymenoptera_data/"
    })

    st.subheader("Transforms & Augmentation")
    st.write({
        "Resize": 256,
        "Center Crop": 224,
        "Normalization Mean": [0.485, 0.456, 0.406],
        "Normalization Std": [0.229, 0.224, 0.225],
        "Augmentations": "Random horizontal flip and random center crop"
    })

    st.subheader("Model Architecture")
    st.write({
        "Base Model": "ResNet18 (pretrained on ImageNet)",
        "Layers Unfrozen": "Final fc layer only",
        "Input Size": "3 x 224 x 224",
    })

    st.subheader("Deployment Information")
    st.write({
        "Model": MODEL_NAME,
        "Framework": "Streamlit (no API)",
        "Docker": "not used",
    })
    st.write("Inspiration from: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html")

def create_app(model_name):
    menu = st.sidebar.radio("Navigation", ["Prediction", "Notes"])
    if menu == "Prediction":
        prediction_tab(model_name)
    elif menu == "Notes":
        logs_tab()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name", default=MODEL_NAME)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    create_app(model_name=args.model)
