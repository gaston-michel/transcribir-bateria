import streamlit as st
from session_state import SessionState
from file_uploader import FileUploader
from ui_components import UIComponents
from separation import separate_drums
from transcription import transcribe

def main():
    st.set_page_config("Transcriptor de batería", ":musical_score:", layout="centered")
    st.title("Transcripción de audio :drum_with_drumsticks:",)
    UIComponents.set_background()

    SessionState.initialize()
    FileUploader.display()

    if SessionState.is_audio_uploaded():
        st.audio(SessionState.get_audio_file())
        UIComponents.display_configuration_options()

        if st.button("Generar transcripción"):
            if not SessionState.has_audio_been_separated():
                with st.spinner('Separando batería...'):
                    separate_drums()

            with st.spinner('Generando archivos...'):
                transcribe()
    
    if SessionState.is_transcription_ready():
        st.divider()
        st.header("Archivos generados")
        UIComponents.display_download_buttons()

if __name__ == "__main__":
    main()