import streamlit as st

from audio.loader import load_audio
from audio.onset_detection import detect_onsets
from audio.beat_detection import detect_beats
from audio.beat_alignment import align_beats_with_onsets
from prediction.drum_part_predictor import predict_drum_parts
from notation.sheet_music_generator import SheetMusicGenerator

def transcribe():
    audio = load_audio(st.session_state["drum_audio_file"])

    onsets = detect_onsets(audio)

    beat_times = detect_beats(audio)
    aligned_beats = align_beats_with_onsets(beat_times, onsets)

    drum_parts = predict_drum_parts(audio, onsets)
    
    sheet_music_files = SheetMusicGenerator.generate_sheet_music(onsets, drum_parts, aligned_beats)

    print("Transcription completed successfully") 
    return sheet_music_files
