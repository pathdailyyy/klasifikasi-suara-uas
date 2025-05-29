import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils import predict
import io
import datetime
import sounddevice as sd
import tempfile
import wavio
import pandas as pd

# Inisialisasi session state untuk history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Judul aplikasi
st.set_page_config(page_title="Sistem Pakar Klasifikasi Suara", layout="centered")
st.title("🧠 Sistem Pakar Klasifikasi Suara Lingkungan")
st.markdown("""
Aplikasi ini merupakan **sistem pakar klasifikasi suara** berbasis **deep learning** yang dikembangkan untuk mengidentifikasi jenis suara dari lingkungan sekitar.

### 🧩 Komponen Sistem Pakar:
- **👨‍🏫 Basis Pengetahuan:** Dataset **ESC-50**
- **🧠 Mesin Inferensi:** Model CNN terlatih
- **🖥️ Antarmuka Pengguna:** Aplikasi Streamlit
---
""")

# --- Penjelasan Kategori Suara ---
st.subheader("📚 Penjelasan Kategori Suara")
with st.expander("Lihat deskripsi kategori suara berdasarkan ESC-50"):
    st.markdown("""
    | Kategori         | Deskripsi Singkat                        |
    |------------------|------------------------------------------|
    | dog              | Suara anjing menggonggong                |
    | rain             | Suara hujan turun                        |
    | fire_crackling   | Suara api menyala                        |
    | clock_tick       | Detak jam                                |
    | sneeze           | Suara orang bersin                       |
    | helicopter       | Suara baling-baling helikopter           |
    | chainsaw         | Suara gergaji mesin                      |
    | crying_baby      | Tangisan bayi                            |
    | rooster          | Kokokan ayam jantan                      |
    | thunderstorm     | Suara badai petir                        |
    """)

# --- Upload File ---
st.subheader("📁 Upload Suara (.wav)")
audio_file = st.file_uploader("Seret dan lepas file di sini", type=["wav"])

# --- Rekam Langsung ---
st.subheader("🎙️ Rekam Suara Langsung")
record = st.button("🔴 Mulai Rekam 3 Detik")

# Prediksi dari file upload
if audio_file:
    st.audio(audio_file, format='audio/wav')

    prediction, mel = predict(audio_file)
    st.success(f"🎯 Kategori Prediksi: **{prediction}**")

    st.session_state['history'].append({
        'file': audio_file.name,
        'prediksi': prediction,
        'waktu': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    st.markdown("### 🎼 Mel-Spectrogram")
    fig, ax = plt.subplots()
    librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=22050, ax=ax)
    ax.set(title='Mel-Spectrogram')
    st.pyplot(fig)

# Prediksi dari hasil rekaman
if record:
    duration = 3  # detik
    fs = 22050

    st.info("⏺️ Merekam selama 3 detik...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("✅ Rekaman selesai!")

    # Simpan ke file sementara
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        safe_audio = np.nan_to_num(recording)
        safe_audio = np.clip(safe_audio, -1.0, 1.0)
        wavio.write(tmpfile.name, safe_audio, fs, sampwidth=2)

        prediction, mel = predict(tmpfile.name)
        st.success(f"🎯 Kategori Prediksi: **{prediction}**")

        st.session_state['history'].append({
            'file': 'rekaman_langsung.wav',
            'prediksi': prediction,
            'waktu': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        st.markdown("### 🎼 Mel-Spectrogram")
        fig, ax = plt.subplots()
        librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=fs, ax=ax)
        ax.set(title='Mel-Spectrogram')
        st.pyplot(fig)

# --- History ---
st.markdown("---")
st.subheader("📜 History Prediksi")

# Filter / Pencarian
search_query = st.text_input("🔍 Cari berdasarkan nama file atau prediksi:")

filtered_history = [h for h in st.session_state['history']
                    if search_query.lower() in h['file'].lower() or search_query.lower() in h['prediksi'].lower()]

if filtered_history:
    for i, h in enumerate(reversed(filtered_history), 1):
        st.markdown(f"**{i}.** 🗂️ *{h['file']}* → 🎯 **{h['prediksi']}** _(🕒 {h['waktu']})_")
else:
    st.info("Belum ada prediksi yang cocok.")
    
    # Tombol download CSV
if st.session_state['history']:
    df_history = pd.DataFrame(st.session_state['history'])
    csv = df_history.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Riwayat sebagai CSV",
        data=csv,
        file_name='riwayat_prediksi.csv',
        mime='text/csv'
    )

if st.button("🗑️ Hapus History"):
    st.session_state['history'] = []
    st.rerun()
