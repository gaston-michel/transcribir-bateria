import os
import shutil
import tempfile
import demucs.separate

def separate_drums(audio_path: str) -> str:
    """
    Processes the given audio file to separate the drum track using Demucs, 
    and saves the separated drum track in a temporary file.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        str: Path to the separated drum track temporary file (drums.wav).
    
    The function calls the Demucs model to split the input audio into two stems: 
    'drums' and 'other'. The separated drum track is saved temporarily.
    """
    # Call Demucs to separate the stems
    demucs.separate.main(["--two-stems", "drums", "-n", "htdemucs", audio_path])

    # Define the directory where Demucs saved the output
    output_dir = os.path.join("separated", "htdemucs", 
                              os.path.splitext(os.path.basename(audio_path))[0])
    drums_path = os.path.join(output_dir, "drums.wav")

    # Create a temporary file for the drum track
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_drums_path = temp_file.name

    # Move the drum track to the temporary file
    shutil.move(drums_path, temp_drums_path)

    # Remove the original directory created by Demucs
    shutil.rmtree("separated")

    # Return the path to the temporary drum track
    return temp_drums_path
