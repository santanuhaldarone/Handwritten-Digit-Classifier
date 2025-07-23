import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
from preprocess import preprocess_mnist_style

model = joblib.load("mnist_knn_model.pkl")

st.title("🧠 MNIST KNN Digit Classifier")
st.markdown("Draw a digit and classify using KNN (trained on MNIST)")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        input_data = preprocess_mnist_style(canvas_result.image_data.astype('uint8'))
        prediction = model.predict(input_data)[0]
        st.header(f"📍 Predicted Digit: {prediction}")
    else:
        st.warning("Please draw something first before predicting.")
