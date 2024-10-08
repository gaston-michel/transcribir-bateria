import base64
import streamlit as st
import os

import config

class UIComponents:
    @staticmethod
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    @staticmethod
    def set_background():  
        bg_css = '''
        <style>
        .stApp > header {
            background-color: transparent;
        }

        .stApp {
            background-color:#11001C;
            background-image:
            radial-gradient(at 68% 87%, hsla(253, 48%, 16%, 1) 0px, transparent 50%),
            radial-gradient(at 63% 78%, hsla(321, 63%, 24%, 1) 0px, transparent 50%),
            radial-gradient(at 68% 25%, hsla(339, 76%, 36%, 1) 0px, transparent 50%),
            radial-gradient(at 5% 58%, hsla(346, 20%, 20%, .5) 0px, transparent 50%),
            radial-gradient(at 93% 6%, hsla(254, 48%, 11%, 1) 0px, transparent 50%);
            background-size: 200% 200%;
            animation: my_animation 10s ease infinite;
        }

        @keyframes my_animation {
            0% {
                background-position: 0% 0%;
            }
            25% {
                background-position: 100% 0%;
            }
            50% {
                background-position: 100% 100%;
            }
            75% {
                background-position: 0% 100%;
            }
            100% {
                background-position: 0% 0%;
            }
        }
        <\style>
        '''
        st.markdown(bg_css, unsafe_allow_html=True)

    @staticmethod
    def display_configuration_options():
        with st.expander("Configuración avanzada"):
            config.FILE_NAME = st.text_input(
                "Nombre del archivo", 
                value=os.path.splitext(st.session_state['audio_file'].name)[0],
                max_chars=30)
            config.START_TIME = st.number_input(
                "Tiempo de inicio (en segundos, dos decimales)", 
                value=config.START_TIME, 
                min_value=0.0, 
                format="%.2f"
            )
            config.TEMPO = st.number_input(
                "Tempo (BPM)", 
                value=float(config.TEMPO) if config.TEMPO is not None else None,
                min_value=1.0, 
                max_value=300.0, 
                step=1.0, 
                format="%.1f"
            )
            config.TIME_SIGNATURE = st.selectbox(
                "Compás", 
                options=("4/4", "2/2", "2/4", "3/4", "3/8", "6/8", "9/8", "12/8")
            )
            config.ONSET_THRESHOLD = st.slider(
                "Umbral de detección de golpe", 
                min_value=0.01, 
                max_value=0.5, 
                value=config.ONSET_THRESHOLD
            )
            config.PREDICTION_THRESHOLD = st.slider(
                "Umbral de predicción", 
                min_value=0.5, 
                max_value=1.0, 
                value=config.PREDICTION_THRESHOLD
            )

    @staticmethod
    def display_download_buttons():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button("Descargar como LY", 
                            data=st.session_state["ly_file"], 
                            file_name=f"{config.FILE_NAME}.ly")
        with col2:
            st.download_button("Descargar como PDF", 
                            data=st.session_state["pdf_file"], 
                            file_name=f"{config.FILE_NAME}.pdf")
        with col3:
            st.download_button("Descargar como MIDI", 
                            data=st.session_state["midi_file"], 
                            file_name=f"{config.FILE_NAME}.mid")