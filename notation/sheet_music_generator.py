import abjad
import numpy as np
import os
import streamlit as st
import subprocess
import tempfile
from typing import List, Tuple

import config

class DrumPartProcessor:
    @staticmethod
    def remove_false_detections(onset_times: np.ndarray, drum_parts: List[Tuple[str, ...]]) -> Tuple[np.ndarray, List[Tuple[str, ...]]]:
        return zip(*[(time, part) for time, part in zip(onset_times, drum_parts) if part != ('None',)])

    @staticmethod
    def convert_to_ly_notation(drum_parts: List[Tuple[str, ...]]) -> List[str]:
        mapping = {
            ('CR',): 'cymc',
            ('CR', 'HH'): '<cymc hh>',
            ('CR', 'HH', 'KD'): '<cymc hh bd>',
            ('CR', 'HH', 'KD', 'SD'): '<cymc hh bd sn>',
            ('CR', 'HH', 'SD'): '<cymc hh sn>',
            ('CR', 'KD'): '<cymc bd>',
            ('CR', 'KD', 'SD'): '<cymc bd sn>',
            ('CR', 'SD'): '<cymc sn>',
            ('HH',): 'hh',
            ('HH', 'KD'): '<hh bd>',
            ('HH', 'KD', 'SD'): '<hh bd sn>',
            ('HH', 'SD'): '<hh sn>',
            ('KD',): 'bd',
            ('KD', 'SD'): '<bd sn>',
            ('SD',): 'sn',
            ('None',): 'r',
        }
        return [mapping.get(drum, 'r') for drum in drum_parts]

class RhythmAnalyzer:
    @staticmethod
    def count_onsets_between_beats(onset_times: np.ndarray, beats: np.ndarray) -> np.ndarray:
        extended_beats = np.concatenate([[-np.inf], beats, [np.inf]])
        onset_counts = []
        for start, end in zip(extended_beats[:-1], extended_beats[1:]):
            count = np.sum((start <= onset_times) & (onset_times < end))
            onset_counts.append(int(count))
        return onset_counts

    @staticmethod
    def determine_musical_figure(onset_counts: np.ndarray) -> List[int]:
        return [figure for count in onset_counts for figure in ([0] if count == 0 else [count * 4] * count)]
        
class LilyPondConverter:
    @staticmethod
    def add_silences(ly_drum_parts: List[str], musical_figures: List[int]) -> List[str]:
        for i in range(len(musical_figures)):
            if musical_figures[i] == 0:
                ly_drum_parts.insert(i, 'r')
        return ly_drum_parts

    @staticmethod
    def create_ly_str(drum_parts: List[str], musical_figures: List[int], time_signature: str) -> str:
        def is_power_of_two(x):
            return x > 0 and (x & (x - 1)) == 0

        notes = []
        i = 0
        while i < len(drum_parts):
            part = drum_parts[i]
            figure = musical_figures[i]
            
            if figure == 0:
                notes.append("r4")
            elif is_power_of_two(figure):
                notes.append(f"{part}{figure}")
            else:
                cant_notes = int(figure / 4)
                approx_figure = 2 * 2 ** (figure // 16)
                tuplet_parts = " ".join(f"{drum_parts[i+j]}{approx_figure * 4}" for j in range(cant_notes))
                tuplet_str = f"\\tuplet {cant_notes}/{approx_figure} {{ {tuplet_parts} }}"
                notes.append(tuplet_str)
                i += cant_notes - 1
            
            i += 1
        
        return r"\drums { \stemUp " + " ".join(notes) + r"} {\time " + time_signature + "}"

class SheetMusicGenerator:
    @staticmethod
    def save_as_pdf(ly_path: str, output_path: str, file_name: str) -> str:
        with open(ly_path, 'r') as file:
            content = file.read()
        
        lines = content.splitlines()
        header_block = f"""##(set-global-staff-size 16)

        \\layout {{ indent = #0 }}

        \\header {{ title = \\markup "{file_name}" }}
        """
        lines.insert(0, header_block)
        modified_content = "\n".join(lines)

        with open(ly_path, 'w') as file:
            file.write(modified_content)
        
        subprocess.run(['lilypond', '-o', output_path, ly_path])
        pdf_path = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(os.path.basename(ly_path))[0]}.pdf")

        return pdf_path

    @staticmethod
    def save_as_midi(ly_path: str, output_path: str, time_signature: str, tempo: int) -> str:
        with open(ly_path, 'r') as file:
            content = file.read()
        
        lines = content.splitlines()
        midi_block = f"\\midi {{\n  \\tempo {time_signature[-1]} = {int(tempo)}\n}}"
        lines.insert(-1, midi_block)
        modified_content = "\n".join(lines)

        with open(ly_path, 'w') as file:
            file.write(modified_content)
        
        subprocess.run(['lilypond', '-o', output_path, ly_path])
        midi_path = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(os.path.basename(ly_path))[0]}.midi")

        return midi_path

    @staticmethod
    def generate_sheet_music(onset_times: np.ndarray, drum_parts: List[Tuple[str, ...]], beats: np.ndarray):
        """
        Generate sheet music for drum parts based on onset times and beats.

        This function processes the input drum parts and timing information to create
        sheet music in LilyPond format. It also generates PDF and MIDI files for the music.

        Args:
            onset_times (np.ndarray): Array of onset times for drum hits in seconds.
            drum_parts (List[Tuple[str, ...]]): List of tuples representing the drum parts played at each onset time. 
            Each tuple contains strings representing individual drum components.
            beats (np.ndarray): Array of beat times in seconds.

        Returns:
            None

        Side Effects:
            - Creates temporary .ly, .pdf, and .midi files.
            - Stores the content of these files in st.session_state under keys "ly_file", 
            "pdf_file", and "midi_file" respectively.
            - Deletes the temporary files after storing their contents.
        """
        filtered_onsets, filtered_parts = DrumPartProcessor.remove_false_detections(onset_times, drum_parts)
        ly_drum_parts = DrumPartProcessor.convert_to_ly_notation(filtered_parts)
        onset_counts = RhythmAnalyzer.count_onsets_between_beats(filtered_onsets, beats)
        musical_figures = RhythmAnalyzer.determine_musical_figure(onset_counts)
        ly_drum_parts = LilyPondConverter.add_silences(ly_drum_parts, musical_figures)
        ly_str = LilyPondConverter.create_ly_str(ly_drum_parts, musical_figures, config.TIME_SIGNATURE)
        
        notation = abjad.Container()
        abjad.attach(abjad.LilyPondLiteral(ly_str), notation)
        score = abjad.Score([notation])
        
        with tempfile.NamedTemporaryFile(suffix='.ly', delete=False) as temp_ly:
            abjad.persist.as_ly(score, temp_ly.name)
            temp_pdf_path = SheetMusicGenerator.save_as_pdf(temp_ly.name, tempfile.gettempdir(), config.FILE_NAME)
            temp_midi_path = SheetMusicGenerator.save_as_midi(temp_ly.name, tempfile.gettempdir(), config.TIME_SIGNATURE, config.TEMPO)

        st.session_state["ly_file"] = open(temp_ly.name, "rb").read()
        st.session_state["pdf_file"] = open(temp_pdf_path, "rb").read()
        st.session_state["midi_file"] = open(temp_midi_path, "rb").read()

        os.remove(temp_ly.name)
        os.remove(temp_pdf_path)
        os.remove(temp_midi_path)