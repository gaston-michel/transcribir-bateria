import os
import tempfile
import shutil
import streamlit as st
from demucs import separate

def separate_drums():
    """Separa la pista de batería del audio original y la almacena en la sesión."""
    temp_audio_path = save_uploaded_audio_to_temp_file()
    
    run_demucs_separation(temp_audio_path)
    
    separated_drum_data = extract_separated_drum_audio(temp_audio_path)
    
    store_separated_drum_audio(separated_drum_data)
    
    cleanup_temp_files(temp_audio_path)

def save_uploaded_audio_to_temp_file():
    """Guarda el audio subido en un archivo temporal y devuelve la ruta."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(st.session_state["audio_file"].getbuffer())
        return temp_audio_file.name

def run_demucs_separation(audio_path):
    """Ejecuta la separación de Demucs en el archivo de audio dado."""
    separate.main(["--two-stems", "drums", "-n", "htdemucs", audio_path])

def extract_separated_drum_audio(original_audio_path):
    """Extrae los datos de audio de batería separados y los devuelve."""
    base_name = os.path.splitext(os.path.basename(original_audio_path))[0]
    separated_dir = os.path.join("separated", "htdemucs", base_name)
    
    if not os.path.exists(separated_dir):
        raise FileNotFoundError(f"El directorio de separación no existe: {separated_dir}")
    
    separated_drum_file = os.path.join(separated_dir, "drums.wav")
    
    with open(separated_drum_file, "rb") as f:
        return f.read()

def store_separated_drum_audio(drum_audio_data):
    """Almacena los datos de audio de batería separados en la sesión de Streamlit."""
    st.session_state["drum_audio_file"] = drum_audio_data

def cleanup_temp_files(temp_audio_path):
    """Limpia los archivos temporales creados durante el proceso de separación."""
    os.remove(temp_audio_path)
    shutil.rmtree("separated", ignore_errors=True)