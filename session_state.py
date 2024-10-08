import streamlit as st

class SessionState:
    @staticmethod
    def initialize():
        st.session_state.setdefault("audio_file", None)
        st.session_state.setdefault("drum_audio_file", None)
        st.session_state.setdefault("ly_file", None)
        st.session_state.setdefault("pdf_file", None)
        st.session_state.setdefault("midi_file", None)

    @staticmethod
    def is_audio_uploaded():
        return st.session_state["audio_file"] is not None

    @staticmethod
    def has_audio_been_separated():
        return st.session_state["drum_audio_file"] is not None

    @staticmethod
    def is_transcription_ready():
        return (
            st.session_state.get("ly_file") is not None and
            st.session_state.get("pdf_file") is not None and
            st.session_state.get("midi_file") is not None
        )

    @staticmethod
    def get_audio_file():
        return st.session_state.get("audio_file")

    @staticmethod
    def get_drum_audio_file():
        return st.session_state.get("drum_audio_file")
    
    @staticmethod
    def store_drum_audio_file(drum_audio_data):
        st.session_state["drum_audio_file"] = drum_audio_data