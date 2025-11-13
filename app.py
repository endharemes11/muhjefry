
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import base64 # Import base64 for embedding image in HTML

# Load the trained YOLOv8 model
# Make sure the model file exists in the correct path
model_path = 'best_safe.pt'
if not os.path.exists(model_path):
    st.error(f"Deteksi Objek Mineral Sedimen Klastik dengan YOLOv8")
else:
    model = YOLO(model_path)
    st.title("Deteksi Objek Mineral Sedimen Klastik dengan YOLOv8")
    
    # Mapping nama kelas berdasarkan indeks dari model /content/best (2).pt
    class_names = {0: 'feldspar', 1: 'kuarsa', 2: 'litik', 3: 'opaq', 4: 'plagioklas'}
    
    uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

        # Perform inference
        results = model(image)

        # Display the results (image with bounding boxes)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            st.image(im, caption="Hasil Deteksi", use_column_width=True)
            
            # Hitung jumlah mineral yang terdeteksi secara keseluruhan
            num_detected = len(r.boxes)
            
            st.write(f"Jumlah mineral yang terdeteksi secara keseluruhan: {num_detected}")
                    
            # --- Add functionality to save the result image as HTML ---
            # Save the result image temporarily
            result_image_path = "result_image.png"
            im.save(result_image_path)

            # Read the image file and encode it in base64
            with open(result_image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()

            # Create simple HTML content with the embedded image
            # Correctly escape the inner f-string
            html_content = f'''
            <!DOCTYPE html>
            <html>
            <head>
            <title>Object Detection Result</title>
            </head>
            <body>
            <h1>Hasil Deteksi Objek</h1>
            <img src="data:image/png;base64,{encoded_string}" alt="Detected Image">
            </body>
            </html>
            '''

            # Provide a download link for the HTML file
            st.download_button(
                label="Unduh Hasil Deteksi (HTML)",
                data=html_content,
                file_name="deteksi_objek_result.html",
                mime="text/html"
            )
            # --- End of added functionality ---

        # Optional: Display prediction details
        # st.write("Detail Prediksi:")
        # for r in results:
        #     for box in r.boxes:
        #         st.write(f"Class: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}, Box: {box.xyxy[0].tolist()}")
