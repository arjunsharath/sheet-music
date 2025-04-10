import os
import numpy as np
import librosa
import librosa.display
import abjad
import subprocess

def check_lilypond():
    """Verify LilyPond installation and return its path"""
    possible_paths = [
        r"C:\lilypond-2.24.4\bin",
        r"C:\Program Files\LilyPond\usr\bin",
        r"C:\LilyPond\usr\bin"
    ]
    for path in possible_paths:
        lilypond_exe = os.path.join(path, "lilypond.exe")
        if os.path.exists(lilypond_exe):
            os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
            return True
    try:
        subprocess.run(['lilypond', '--version'], capture_output=True)
        return True
    except FileNotFoundError:
        return False

def note_to_code(note_name: str) -> str:
    note_map = {
        'C': 'a', 'C#': 'b', 'Db': 'b',
        'D': 'c', 'D#': 'd', 'Eb': 'd',
        'E': 'e', 'F': 'f', 'F#': 'g', 'Gb': 'g',
        'G': 'h', 'G#': 'i', 'Ab': 'i',
        'A': 'j', 'A#': 'k', 'Bb': 'k',
        'B': 'l'
    }
    base_note = ''.join(c for c in note_name if c.isalpha() or c in ['#', 'b'])
    octave = ''.join(c for c in note_name if c.isdigit()) or '4'
    return f"{note_map.get(base_note, 'a')},{octave}"

def determine_note_duration(onset_time, next_onset_time, tempo):
    if next_onset_time is None:
        return 1
    duration_in_seconds = next_onset_time - onset_time
    beats_duration = (duration_in_seconds * tempo) / 60
    if beats_duration >= 3:
        return 4
    elif beats_duration >= 1.5:
        return 2
    elif beats_duration >= 0.75:
        return 1
    else:
        return 0.5

def detect_notes_in_song(filename):
    y, sr = librosa.load(filename)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    notes_sequence = []
    for i, onset_time in enumerate(onset_times):
        next_onset_time = onset_times[i + 1] if i < len(onset_times) - 1 else None
        duration = determine_note_duration(onset_time, next_onset_time, tempo)
        segment = y[int(onset_time * sr):int(next_onset_time * sr)] if next_onset_time else y[int(onset_time * sr):]
        if len(segment) < 512:
            continue
        fft_result = np.fft.fft(segment * np.hanning(len(segment)))
        fft_freqs = np.fft.fftfreq(len(fft_result), 1/sr)
        frequency = abs(fft_freqs[np.argmax(np.abs(fft_result[:len(fft_result)//2]))])
        if frequency > 0:
            midi_number = int(round(69 + 12 * np.log2(frequency / 440.0)))
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_name = note_names[midi_number % 12]
            octave = (midi_number // 12) - 1
            note_code = note_to_code(f"{note_name}{octave}")
            notes_sequence.append(f"{duration},{note_code}")
    return notes_sequence

class SheetMusicGenerator:
    def __init__(self):
        self.letter_to_pitch = {
            'a': 'c', 'b': 'cs', 'c': 'd', 'd': 'ds',
            'e': 'e', 'f': 'f', 'g': 'fs', 'h': 'g',
            'i': 'gs', 'j': 'a', 'k': 'as', 'l': 'b'
        }
        self.duration_map = {4.0: (1, 1), 2.0: (1, 2), 1.0: (1, 4), 0.5: (1, 8)}
    def create_sheet_music(self, notes_data, title="Piano Piece"):
        staff = abjad.Staff()
        for note_str in notes_data:
            duration, note_code, octave = note_str.split(',')
            pitch = f"{self.letter_to_pitch.get(note_code, 'c')}{octave}"
            duration_tuple = self.duration_map.get(float(duration), (1, 4))
            staff.append(abjad.Note(pitch, abjad.Duration(duration_tuple)))
        score = abjad.Score([staff])
        abjad.attach(abjad.Markup(title), staff[0])
        pdf_path = os.path.abspath("piano_sheet_music.pdf")
        abjad.persist.as_pdf(abjad.LilyPondFile(items=[score]), pdf_path)
        return pdf_path

def process_audio_to_sheet(audio_file):
    print("Extracting notes from the audio file...")
    notes = detect_notes_in_song(audio_file)
    if not notes:
        print("No notes detected. Exiting...")
        return
    if not check_lilypond():
        print("LilyPond is required for sheet music generation. Please install it.")
        return
    print("Generating sheet music...")
    generator = SheetMusicGenerator()
    pdf_path = generator.create_sheet_music(notes, "Generated Composition")
    print(f"Sheet music saved as: {pdf_path}")

if __name__ == "__main__":
    audio_file = "C:/Users/arjun/Documents/piano/lala.mp3"
    process_audio_to_sheet(audio_file)
