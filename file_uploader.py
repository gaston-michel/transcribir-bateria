import streamlit as st
import config

class FileUploader:
    @staticmethod
    def display():
        st.file_uploader("Subir archivo de audio", type=["mp3", "wav"], key="audio_uploader", 
                         on_change=FileUploader._file_uploader_callback)

    @staticmethod
    def _file_uploader_callback():
        if FileUploader._has_audio_been_removed() or FileUploader._has_audio_been_replaced():
            st.session_state["drum_audio_file"] = None
            st.session_state["ly_file"] = None
            st.session_state["pdf_file"] = None
            st.session_state["midi_file"] = None
            
            config.FILE_NAME = None
            config.START_TIME = 0.0
            config.TEMPO = None
            config.TIME_SIGNATURE = "4/4"
            config.ONSET_THRESHOLD = 0.1
            config.PREDICTION_THRESHOLD = 0.7
        
        st.session_state["audio_file"] = st.session_state.get("audio_uploader")
    
    @staticmethod
    def _has_audio_been_removed():
        return (st.session_state["audio_file"] is not None and
                st.session_state.get("audio_uploader") is None)
    
    @staticmethod
    def _has_audio_been_replaced():
        return (st.session_state["audio_file"] != st.session_state.get("audio_uploader")
                and st.session_state["audio_file"] is not None)