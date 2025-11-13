import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import base64

# Pastikan model tersedia
model_path = 'best_safe.pt'

if not os.path.exists(model_path):
    st.error("❌ File model tidak ditemukan. Pastikan 'best_safe.pt' ada di direktori yang benar.")
else:
    # Muat model YOLOv8
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
            # Plot hasil deteksi
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            st.image(im, caption="🔍 Hasil Deteksi", use_column_width=True)

            # Hitung jumlah per kelas
            class_counts = {name: 0 for name in class_names.values()}
            total_detected = 0

            if r.boxes is not None and len(r.boxes) > 0:
                classes = r.boxes.cls.cpu().numpy().astype(int)
                for c in classes:
                    class_name = class_names.get(c, "Tidak diketahui")
                    class_counts[class_name] += 1
                    total_detected += 1

            # Total hanya untuk kelas feldspar, kuarsa, litik
            total_selected = (
                class_counts["feldspar"] + class_counts["kuarsa"] + class_counts["litik"]
            )

            # 🔢 Tampilkan hasil di Streamlit
            st.markdown("### 📊 Jumlah dan Persentase Deteksi per Kelas:")

            if total_detected == 0:
                st.warning("Tidak ada objek mineral yang terdeteksi pada gambar ini.")
            else:
                for cls_name, count in class_counts.items():
                    if cls_name in ["feldspar", "kuarsa", "litik"]:
                        percentage = (count / total_selected) * 100 if total_selected > 0 else 0
                        st.write(f"- **{cls_name.capitalize()}**: {count} ({percentage:.2f}%)")
                    else:
                        st.write(f"- **{cls_name.capitalize()}**: {count}")

                st.markdown(f"### 🧮 **Total Semua Mineral Terdeteksi:** {total_detected}")
                st.markdown(f"### ⚗️ **Total (Feldspar + Kuarsa + Litik):** {total_selected}")

            # --- Simpan hasil ke file HTML ---
            result_image_path = "result_image.png"
            im.save(result_image_path)

            with open(result_image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()

            # Buat tabel HTML
            html_rows = ""
            for cls_name, count in class_counts.items():
                if cls_name in ["feldspar", "kuarsa", "litik"]:
                    percentage = (count / total_selected) * 100 if total_selected > 0 else 0
                    html_rows += f"<tr><td>{cls_name}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
                else:
                    html_rows += f"<tr><td>{cls_name}</td><td>{count}</td><td>-</td></tr>"

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
                        width: 480px;
                    }}
                    th, td {{
                        border: 1px solid #aaa;
                        padding: 8px 12px;
                        text-align: center;
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
                <p><strong>Total Semua Mineral Terdeteksi:</strong> {total_detected}</p>
                <p><strong>Total (Feldspar + Kuarsa + Litik):</strong> {total_selected}</p>
                <table>
                    <tr><th>Kelas Mineral</th><th>Jumlah</th><th>Persentase*</th></tr>
                    {html_rows}
                </table>
                <p style="font-size: 13px; color: #555;">*Persentase hanya dihitung untuk Feldspar, Kuarsa, dan Litik.</p>
                <img src="data:image/png;base64,{encoded_string}" alt="Detected Image">
            </body>
            </html>
            """

            # Tombol unduh hasil HTML
            st.download_button(
                label="⬇️ Unduh Hasil Deteksi (HTML)",
                data=html_content,
                file_name="deteksi_objek_result.html",
                mime="text/html"
            )
