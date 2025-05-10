# import streamlit as st
# import pickle
# import numpy as np
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing import image
# from PIL import Image

# # Load model from .pkl file
# with open("model.pkl", "rb") as f:
#     data = pickle.load(f)

# model_structure = data["model_structure"]
# model_weights = data["model_weights"]

# model = model_from_json(model_structure)
# model.set_weights(model_weights)

# # Define the class names    
# class_names = ['Diabetic Retinopathy', 'Glaucoma', 'Healthy']  # Update as needed

# # Streamlit UI
# st.title("Retinal Disease Classifier")
# st.write("Upload a fundus image and the model will predict the disease.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     # Preprocess image
#     img = img.resize((224, 224))  # Match your training size
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Predict
#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]

#     st.markdown(f"### 🧠 Prediction: **{predicted_class}**")







#   SECOND MODEL


# import streamlit as st
# import numpy as np
# import pickle
# from PIL import Image
# import matplotlib.pyplot as plt
# import os


# # Page config
# st.set_page_config(page_title="Retinal Disease Detector", layout="wide")
# st.title("🧠 Retinal Disease Detection Using Deep Learning")
# st.markdown("""
# Upload one or more **retinal fundus images**, and the model will predict whether the image shows signs of **Diabetic Retinopathy**, **Glaucoma**, or if it is **Healthy**.
# """)


# # Load model
# @st.cache_resource
# def load_model():
#     model = pickle.load(open("model.pkl", "rb"))
#     return model

# model = load_model()
# labels = ['Diabetic Retinopathy', 'Glaucoma', 'Healthy']



# st.sidebar.title("🔍 Model Info")
# st.sidebar.markdown("""
# - **Model**: CNN
# - **Input Size**: 224x224
# - **Classes**: Diabetic Retinopathy, Glaucoma, Healthy
# - **Framework**: TensorFlow/Keras
# """)

# # File uploader for multiple files
# uploaded_files = st.file_uploader("Upload one or more retinal images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         st.markdown("---")
#         col1, col2 = st.columns([1, 2])

#         # Load and display image
#         image = Image.open(uploaded_file).convert("RGB")
#         col1.image(image, caption=uploaded_file.name, use_column_width=True)

#         # Preprocessing
#         img = image.resize((224, 224))
#         img = np.array(img) / 255.0
#         img = np.expand_dims(img, axis=0)

#         # Prediction
#         prediction = model.predict(img)
#         predicted_class = labels[np.argmax(prediction)]
#         confidence = np.max(prediction)

#         # Display results
#         col2.subheader("🧪 Prediction Result")
#         col2.write(f"**Class:** {predicted_class}")
#         col2.write(f"**Confidence:** {confidence * 100:.2f}%")

#         # Visualize prediction probabilities
#         col2.subheader("📊 Confidence Breakdown")
#         fig, ax = plt.subplots()
#         ax.bar(labels, prediction[0], color=['orange', 'red', 'green'])
#         ax.set_ylabel("Probability")
#         ax.set_ylim([0, 1])
#         st.pyplot(fig)

#         # Placeholder Grad-CAM
#         col2.subheader("🔍 Model Attention (Placeholder)")
#         col2.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Color_fundus_photograph_of_normal_right_eye.jpg/640px-Color_fundus_photograph_of_normal_right_eye.jpg",
#                    caption="Example Grad-CAM Heatmap", use_column_width=True)

# # Evaluation plots
# st.markdown("---")
# st.subheader("📈 Model Training Summary (Placeholder)")
# fig2, ax2 = plt.subplots()
# epochs = [1, 2, 3, 4, 5]
# acc = [0.6, 0.72, 0.78, 0.82, 0.86]
# loss = [1.2, 0.9, 0.6, 0.5, 0.4]
# ax2.plot(epochs, acc, label="Accuracy", marker='o')
# ax2.plot(epochs, loss, label="Loss", marker='x')
# ax2.set_xlabel("Epoch")
# ax2.set_title("Training Accuracy & Loss")
# ax2.legend()
# st.pyplot(fig2)

# st.markdown("---")
# st.markdown("Made with ❤️ by **Jay** | Deep learning Project")




#  THIRD MODEL




import streamlit as st
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO
from fpdf import FPDF
from tensorflow.keras.models import model_from_json

st.set_page_config(page_title="Retinal Disease Detector", layout="wide")
# Load model


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)

    model = model_from_json(data["model_structure"])
    model.set_weights(data["model_weights"])
    return model


# @st.cache_resource
# def load_model():
#     model = pickle.load(open("model.pkl", "rb"))
#     return model

model = load_model()
labels = ['Diabetic Retinopathy', 'Glaucoma', 'Healthy']

# Page config

st.title("👁️ Retinal Disease Detection Using Deep Learning")
st.markdown("""
Upload one or more **Retinal Fundus Images**, and the model will predict whether the image shows signs of **Diabetic Retinopathy**, **Glaucoma**, or if it is **Healthy**.
""")

st.sidebar.title("🔍 Model Info")
st.sidebar.markdown("""
- **Model**: CNN
- **Input Size**: 224x224
- **Classes**: Diabetic Retinopathy, Glaucoma, Healthy
- **Framework**: TensorFlow/Keras
""")

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload one or more retinal images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

report_data = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        col1.image(image, caption=uploaded_file.name, use_column_width=True)

        # Preprocessing
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img)
        predicted_class = labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display results
        col2.subheader("🧪 Prediction Result")
        
        # col2.write(f"**Class:** {predicted_class }")
        if predicted_class == "Healthy":
            col2.success(f"🟢 Class: {predicted_class}")
        else:
            col2.error(f"🔴 Class: {predicted_class}")
        col2.write(f"**Confidence:** {confidence * 100:.2f}%")

        # Visualize prediction probabilities
        # col2.subheader("📊 Confidence Breakdown")
        # fig, ax = plt.subplots()
        # ax.bar(labels, prediction[0], color=['orange', 'red', 'green'])
        # ax.set_ylabel("Probability")
        # ax.set_ylim([0, 1])
        # st.pyplot(fig)

        col2.subheader("📊 Confidence Breakdown")
        fig, ax = plt.subplots(figsize=(4, 2))  # smaller figure
        bar_width = 0.3  # slimmer bars
        ax.bar(labels, prediction[0], color=['orange', 'red', 'green'], width=bar_width)
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        st.pyplot(fig)


        # Save figure to buffer
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)

        # Collect data for report
        report_data.append({
            "filename": uploaded_file.name,
            "class": predicted_class,
            "confidence": confidence,
            "img": image,
            "chart": buffer.read()
        })

# PDF Report Generation
if report_data:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for entry in report_data:
        pdf.add_page()

        pdf.set_text_color(0, 0, 128)
        pdf.set_font("Arial", style='B', size=18)
        pdf.cell(200, 10, txt="Retinal Disease Detection Report", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Filename: {entry['filename']}", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Class: {entry['class']}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {entry['confidence'] * 100:.2f}%", ln=True)
        pdf.ln(5)

        if entry['class'] == "Diabetic Retinopathy":
            pdf.multi_cell(0, 10, "Diabetic Retinopathy is a complication of diabetes that affects the eyes. "
                                  "It is caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). "
                                  "Early detection and treatment can help prevent vision loss.")
        elif entry['class'] == "Glaucoma":
            pdf.multi_cell(0, 10, "Glaucoma is a group of eye conditions that damage the optic nerve, which is vital for good vision. "
                                  "This damage is often caused by abnormally high pressure in your eye. "
                                  "It is one of the leading causes of blindness for people over the age of 60.")
        else:
            pdf.multi_cell(0, 10, "No signs of disease detected. The retina appears healthy and normal based on the prediction model.")

        # Save and embed prediction chart
        chart_path = f"temp_chart_{entry['filename'].replace('.', '_')}.png"
        with open(chart_path, "wb") as f:
            f.write(entry['chart'])
        page_width = pdf.w - 2 * pdf.l_margin  # full usable width of the page
        pdf.image(chart_path, x=pdf.l_margin, w=page_width)
        os.remove(chart_path)           

    pdf_output = "retinal_report.pdf"
    pdf.output(pdf_output)

    # Offer PDF download early
    with open("retinal_report.pdf", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    # href = f'<a href="data:application/octet-stream;base64,{b64}" download="retinal_report.pdf">📄 Download Medical Report (PDF)</a>'
    # col2.markdown(href, unsafe_allow_html=True)
    button_html = f"""
    <div style="text-align:center; margin-top: 10px;">
        <a href="data:application/octet-stream;base64,{b64}" 
            download="retinal_report.pdf" 
            style="
                display: inline-block;
                background-color: #007BFF;
                color: white;
                padding: 12px 24px;
                font-size: 16px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                ">
            📄 Download Medical Report (PDF)
            </a>
        </div>
        """
    col2.markdown(button_html, unsafe_allow_html=True)




    # Placeholder Grad-CAM
    col2.subheader("🔍 Model Attention (Placeholder)")
    col2.image("https://cdn.pixabay.com/photo/2024/02/06/16/09/ai-generated-8557387_1280.jpg",
                caption="Example Grad-CAM Heatmap",  width=200 )

# Evaluation plots
# st.markdown("---")
# st.subheader("📈 Model Training Summary (Placeholder)")
# fig2, ax2 = plt.subplots()
# epochs = [1, 2, 3, 4, 5]
# acc = [0.6, 0.72, 0.78, 0.82, 0.86]
# loss = [1.2, 0.9, 0.6, 0.5, 0.4]
# ax2.plot(epochs, acc, label="Accuracy", marker='o')
# ax2.plot(epochs, loss, label="Loss", marker='x')
# ax2.set_xlabel("Epoch")
# ax2.set_title("Training Accuracy & Loss")
# ax2.legend()
# st.pyplot(fig2)


# Class distribution mock visualization
st.markdown("---")
st.subheader("📊 Class Distribution in Training Data (Mock Data)")

class_counts = {'Diabetic Retinopathy': 8000, 'Glaucoma': 4000, 'Healthy': 6000}
fig3, ax3 = plt.subplots(figsize=(3.5, 2.5))
ax3.bar(class_counts.keys(), class_counts.values(), color=['orange', 'red', 'green'], width=0.35)
ax3.set_ylabel("Image Count")
ax3.set_title("Distribution of Classes in Training Set")
ax3.tick_params(axis='x', labelsize=9)
st.pyplot(fig3)




st.markdown("---")
st.markdown("Made with ❤️ by **Jay** | Deep learning Project")
