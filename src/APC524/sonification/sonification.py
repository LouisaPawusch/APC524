from email.mime import audio
import numpy as np
import simpleaudio as sa
from time import sleep
from scipy.io.wavfile import write as wavwrite
from moviepy.editor import VideoFileClip, AudioFileClip

def count_living_cells(grid):
    """
    Counts the number of living cells in a cellular automaton grid.

    Parameters
    ----------
    grid : np.ndarray
        2D array representing the cellular automaton grid.

    Returns
    -------
    int
        Number of living cells (cells with value 1).
    """
    return np.sum(grid > 0)


def map_count_to_freq(count, grid_size, scale=None):
    """
    Map the number of living cells to an audio frequency (Hz).

    The mapping uses a logistic function centered on the typical
    cell count range to emphasize musical variation where activity
    is most dynamic. The resulting frequency is quantized to the
    nearest note in the provided scale.

    Parameters
    ----------
    count : int
        Number of living cells.
    grid_size : int
        Total number of cells in the grid.
    scale : list of float, optional
        Multiplicative frequency ratios defining a musical scale.
        Defaults to a 3-octave C-major scale with 36 notes.

    Returns
    -------
    float
        Frequency in Hz corresponding to the current automaton state.
    """
    if scale is None:
        scale = [2 ** (i / 12) for i in range(36)]

    center, spread = grid_size / 9, grid_size / 36
    # compress values into [0,1] with most variation around center
    x = 1 / (1 + np.exp(-(count - center) / spread))

    # Map to scale indices
    index = int(x * (len(scale) - 1))
    base_freq = 220  # A3
    freq = base_freq * scale[index]
    return round(freq, 1)

def make_tone(frequency, duration=0.2, volume=0.3, sample_rate=44100):
    """
    Generates a tone at the given frequency.

    Parameters
    ----------
    frequency : int
        Frequency of the tone in Hz.
    duration : float, optional
        Duration of the tone in seconds, de
    volume : float, optional
        Volume of the tone (0.0 to 1.0), defaults to 0.3.
    sample_rate : int
        Sample rate for audio playback, defaults to 44100 Hz.

    Returns
    -------
    np.ndarray of shape (duration * sample_rate,), dtype=int16
        Array of PCM-encoded audio samples.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = tone * (32767 * volume)
    audio = audio.astype(np.int16)
    return audio


def sonify_automaton(automaton, interval=200, save_audio_as=None):
    """
    Convert a cellular automaton's evolution into a sequence of tones.

    The number of living cells at each step is mapped to a frequency,
    and the resulting tones are concatenated into an audio waveform.
    Optionally saves the combined audio as a .wav file.

    Parameters
    ----------
    automaton : CellularAutomaton
        The cellular automaton instance with history attribute.
    interval : int, optional
        Time in milliseconds between steps, defaults to 200.
    save_audio_as : str or None, optional
        Path to save the generated audio waveform (.wav). If None,
        the audio is not written to disk.

    Returns
    -------
    None

    Raises 
    ------
    ValueError
        If automaton is None.
    ValueError
        If automaton is None.
    """
    if automaton is None:
        err_msg = "Automaton instance must be provided"
        raise ValueError(err_msg)
    
    total_cells = automaton.history[0].size
    print(f"Total cells in grid: {total_cells}")
    sample_rate = 44100
    audio_concat = np.array([], dtype=np.int16)

    print("🎵 Starting Game of Life sonification...")

    for step, grid in enumerate(automaton.history):
        count = count_living_cells(grid)
        freq = map_count_to_freq(count, total_cells)
        duration = interval / 1000.0
        print(f"Step {step:03d} | Alive cells: {count} | Freq: {freq:.1f} Hz")

        tone_audio = make_tone(freq, duration=duration, sample_rate=sample_rate)
        audio_concat = np.concatenate((audio_concat, tone_audio))

    if save_audio_as:
        wavwrite(save_audio_as, sample_rate, audio_concat)
        print(f"💾 Saved audio to {save_audio_as}")

    print("🎶 Done sonifying automaton.")


def merge_audio_video(video_path, audio_path, output_path):
    """
    Merges an audio file with a video file.

    Parameters
    ----------
    video_path : str
        Path to the video file (.gif or .mp4).
    audio_path : str
        Path to the audio file (.wav).
    output_path : str
        Path to save the merged output video file (.mp4).

    Returns
    -------
    None

    Notes
    -----
    Uses MoviePy to multiplex the video and audio streams. The resulting
    file is encoded with H.264 video and AAC audio codecs.
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, logger=None)

    print(f"✅ Merged audio and video saved to {output_path}")