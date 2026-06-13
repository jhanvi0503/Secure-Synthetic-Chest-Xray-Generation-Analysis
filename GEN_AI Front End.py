import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="X-Ray AI Analyzer",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Remove top spacing */
.block-container {
    padding-top: 2rem;
}

/* Title */
.title {
    text-align: center;
    font-size: 70px;
    font-weight: bold;
    color: #00e6e6;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 24px;
    color: #d9d9d9;
    margin-bottom: 50px;
}

/* Upload section */
.upload-box {
    border: 3px dashed #00e6e6;
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    background-color: rgba(255,255,255,0.05);
}

/* Prediction Result */
.result {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #00ffcc;
    margin-top: 30px;
}

/* Button Styling */
.stButton>button {
    width: 100%;
    height: 65px;
    border-radius: 15px;
    font-size: 22px;
    font-weight: bold;
    background-color: #00e6e6;
    color: black;
    border: none;
}

.stButton>button:hover {
    background-color: #00b3b3;
    color: white;
}

/* Hide streamlit footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# TITLE
# ---------------------------------
st.markdown(
    '<div class="title">🩺 Secure Synthetic X-Ray AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Based Pneumonia Detection System</div>',
    unsafe_allow_html=True
)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load("resnet50.pth", map_location="cpu")
    )

    model.eval()

    return model

model = load_model()

# ---------------------------------
# IMAGE TRANSFORM
# ---------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------------------------
# CLASS LABELS
# ---------------------------------
class_names = ["NORMAL", "PNEUMONIA"]

# ---------------------------------
# LAYOUT
# ---------------------------------
col1, col2 = st.columns(2)

uploaded_file = None

with col1:

    st.markdown(
        '<div class="upload-box"><h2>📤 Upload Chest X-Ray</h2></div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"]
    )

with col2:

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")

        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

# ---------------------------------
# PREDICTION
# ---------------------------------
if uploaded_file is not None:

    if st.button("🔍 Analyze X-Ray"):

        input_img = transform(image).unsqueeze(0)

        with torch.no_grad():

            output = model(input_img)

            _, pred = torch.max(output, 1)

        prediction = class_names[pred.item()]

        st.markdown(
            f'<div class="result">Prediction: {prediction}</div>',
            unsafe_allow_html=True
        )