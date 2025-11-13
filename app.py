import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import base64

# Path model
model_path = 'best_safe.pt'

if not os.path.exists(model_path):
    st.error("File model tidak ditemukan. Pastikan 'best_safe.pt' ada di direktori yang benar.")
else:
    model = YOLO(model_path)
    st.title("🪨 Deteksi Objek Mineral Sedimen Klastik dengan YOLOv8")

    # Mapping nama kelas
    class_names = {0: 'feldspar', 1: 'kuarsa', 2: 'litik', 3: 'opaq', 4: 'plagioklas'}

    uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Gambar yang Diunggah", use_column_width=True)

        # Inference
        results = model(image)

        # Tampilkan hasil deteksi
        for r in results:
            im_array = r.plot()  # array hasil deteksi
            im = Image.fromarray(im_array[..., ::-1])  # konversi ke RGB
            st.image(im, caption="🔍 Hasil Deteksi", use_column_width=True)

            # Hitung jumlah total objek
            total_detected = len(r.boxes)
            st.write(f"**Jumlah total mineral yang terdeteksi:** {total_detected}")

            # --- Hitung jumlah per kelas ---
            class_counts = {name: 0 for name in class_names.values()}

            # Ambil prediksi kelas dari hasil YOLO
            if r.boxes is not None and len(r.boxes) > 0:
                classes = r.boxes.cls.cpu().numpy().astype(int)
                for c in classes:
                    class_name = class_names.get(c, "Tidak diketahui")
                    class_counts[class_name] += 1

            # Tampilkan hasil per kelas
            st.write("### Jumlah Deteksi per Kelas:")
            for cls_name, count in class_counts.items():
                st.write(f"- **{cls_name.capitalize()}**: {count}")

            # --- Simpan hasil ke file HTML ---
            result_image_path = "result_image.png"
            im.save(result_image_path)

            with open(result_image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <title>Hasil Deteksi Mineral</title>
            <meta charset="UTF-8">
            </head>
            <body style="font-family: Arial; margin: 20px;">
                <h1>Hasil Deteksi Objek Mineral</h1>
                <p><strong>Jumlah total:</strong> {total_detected}</p>
                <ul>
                    {''.join([f'<li>{cls}: {count}</li>' for cls, count in class_counts.items()])}
                </ul>
                <img src="data:image/png;base64,{encoded_string}" alt="Detected Image" style="max-width: 100%;">
            </body>
            </html>
            """

            st.download_button(
                label="⬇️ Unduh Hasil Deteksi (HTML)",
                data=html_content,
                file_name="deteksi_objek_result.html",
                mime="text/html"
            )
