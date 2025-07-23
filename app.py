import streamlit as st
from streamlit_drawable_canvas import st_canvas
from model import KNNModelHandler
from preprocess import preprocess
from PIL import Image

st.set_page_config(page_title="KNN Explorer", layout="centered")
st.title("🧠 KNN Explorer - Handwritten Digit Classifier")
st.markdown("Draw a digit (0-9) in the canvas below and click **Predict** to classify it.")

@st.cache_resource
def load_model():
    return KNNModelHandler()

model = load_model()

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    width=192,
    height=192,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("🔍 Predict"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
        img_array = preprocess(image)
        pred, conf = model.predict(img_array)
        st.success(f"Predicted Digit: **{pred}**\nConfidence: **{conf:.2f}**")
    else:
        st.warning("✏️ Please draw a digit first!")

if st.button("🔄 Clear"):
    st.experimental_rerun()

st.caption("📊 Model trained on sklearn's digits dataset (8x8 grayscale). Built with ❤️ using Streamlit.")
