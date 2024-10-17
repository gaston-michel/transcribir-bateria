import os
import tempfile
import shutil
import streamlit as st
from demucs import separate

def separate_drums():
    """Separates the drum track from the original audio and stores it in the session."""
    temp_audio_path = save_uploaded_audio_to_temp_file()
    
    run_demucs_separation(temp_audio_path)
    
    separated_drum_data = extract_separated_drum_audio(temp_audio_path)
    
    store_separated_drum_audio(separated_drum_data)
    
    cleanup_temp_files(temp_audio_path)

def save_uploaded_audio_to_temp_file():
    """Saves the uploaded audio to a temporary file and returns the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(st.session_state["audio_file"].getbuffer())
        return temp_audio_file.name

def run_demucs_separation(audio_path):
    """Run the Demucs split on the given audio file."""
    separate.main(["--two-stems", "drums", "-n", "htdemucs", audio_path])

def extract_separated_drum_audio(original_audio_path):
    """Extracts the separate drum audio data and returns it."""
    base_name = os.path.splitext(os.path.basename(original_audio_path))[0]
    separated_dir = os.path.join("separated", "htdemucs", base_name)
    
    if not os.path.exists(separated_dir):
        raise FileNotFoundError(f"El directorio de separación no existe: {separated_dir}")
    
    separated_drum_file = os.path.join(separated_dir, "drums.wav")
    
    with open(separated_drum_file, "rb") as f:
        return f.read()

def store_separated_drum_audio(drum_audio_data):
    """Stores the separate drum audio data in the Streamlit session."""
    st.session_state["drum_audio_file"] = drum_audio_data

def cleanup_temp_files(temp_audio_path):
    """Cleans up temporary files created during the separation process."""
    os.remove(temp_audio_path)
    shutil.rmtree("separated", ignore_errors=True)