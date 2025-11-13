import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import base64
import matplotlib.pyplot as plt

# Pastikan model tersedia
model_path = 'best_safe.pt'

if not os.path.exists(model_path):
    st.error("❌ File model tidak ditemukan. Pastikan 'best_safe.pt' ada di direktori yang benar.")
else:
    # Muat model
    model = YOLO(model_path)
    st.title("🪨 Deteksi Objek Mineral Sedimen Klastik dengan YOLOv8")

    # Mapping nama kelas
    class_names = {0: 'feldspar', 1: 'kuarsa', 2: 'litik', 3: 'opaq', 4: 'plagioklas'}

    uploaded_file = st.file_uploader("📤 Unggah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Gambar yang Diunggah", use_column_width=True)

        # Jalankan inferensi YOLO
        results = model(image)

        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            st.image(im, caption="🔍 Hasil Deteksi", use_column_width=True)

            # Hitung deteksi
            class_counts = {name: 0 for name in class_names.values()}
            total_detected = 0

            if r.boxes is not None and len(r.boxes) > 0:
                classes = r.boxes.cls.cpu().numpy().astype(int)
                for c in classes:
                    class_name = class_names.get(c, "Tidak diketahui")
                    class_counts[class_name] += 1
                    total_detected += 1

            # 🔢 Tampilkan hasil di Streamlit
            st.markdown("### 📊 Jumlah Deteksi per Kelas:")
            for cls_name, count in class_counts.items():
                st.write(f"- **{cls_name.capitalize()}**: {count}")

            st.markdown(f"### 🧮 **Total Jumlah Mineral yang Terdeteksi:** {total_detected}")

            # --- Visualisasi Bar Chart ---
            fig, ax = plt.subplots()
            ax.bar(class_counts.keys(), class_counts.values())
            ax.set_xlabel("Kelas Mineral")
            ax.set_ylabel("Jumlah")
            ax.set_title("Distribusi Deteksi Mineral")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # --- Simpan hasil ke file HTML ---
            result_image_path = "result_image.png"
            im.save(result_image_path)

            with open(result_image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()

            # HTML dengan jumlah per kelas dan total
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Hasil Deteksi Mineral</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 30px; }}
                    h1 {{ color: #2c3e50; }}
                    table {{
                        border-collapse: collapse;
                        margin-top: 20px;
                        width: 400px;
                    }}
                    th, td {{
                        border: 1px solid #aaa;
                        padding: 8px 12px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    img {{
                        margin-top: 20px;
                        max-width: 90%;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                    }}
                </style>
            </head>
            <body>
                <h1>Hasil Deteksi Objek Mineral</h1>
                <p><strong>Total Jumlah Mineral yang Terdeteksi:</strong> {total_detected}</p>
                <table>
                    <tr><th>Kelas Mineral</th><th>Jumlah</th></tr>
                    {''.join([f'<tr><td>{cls}</td><td>{count}</td></tr>' for cls, count in class_counts.items()])}
                </table>
                <img src="data:image/png;base64,{encoded_string}" alt="Detected Image">
            </body>
            </html>
            """

            # Tombol unduh hasil
            st.download_button(
                label="⬇️ Unduh Hasil Deteksi (HTML)",
                data=html_content,
                file_name="deteksi_objek_result.html",
                mime="text/html"
            )
