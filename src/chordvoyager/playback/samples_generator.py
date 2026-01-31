import numpy as np
from scipy import signal
from scipy.io import wavfile
import os

# Parameters
SAMPLE_RATE = 44100
DURATION = 3
FADE_OUT_TIME = 0.5
OUTPUT_DIR = "samples"


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def midi_to_freq(midi_note):
    return 440 * (2 ** ((midi_note - 69) / 12))


# Generate sine wave with harmonics
def generate_sine_wave(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Fundamental frequency
    wave = np.sin(2 * np.pi * freq * t) * 0.1
    return wave, t


# Apply lowpass filter
def apply_lowpass_filter(wave, cutoff_freq, sample_rate):
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype="low")
    return wave


# Apply fade out
def apply_fade_out(wave, fade_time, sample_rate):
    fade_samples = int(fade_time * sample_rate)
    fade_curve = np.linspace(1, 0, fade_samples)
    wave[-fade_samples:] *= fade_curve
    return wave


# Generate and save all MIDI notes
for midi_note in range(0, 128):
    freq = midi_to_freq(midi_note)
    wave, t = generate_sine_wave(freq, DURATION, SAMPLE_RATE)
    if midi_note < 125:
        # Apply lowpass filter (cutoff at 2x fundamental to reduce harshness)
        wave = apply_lowpass_filter(wave, freq * 2, SAMPLE_RATE)

    # Apply fade out
    wave = apply_fade_out(wave, FADE_OUT_TIME, SAMPLE_RATE)

    # Normalize to 16-bit range
    wave = np.int16(wave / np.max(np.abs(wave)) * 32767)

    # Save file
    filename = os.path.join(OUTPUT_DIR, f"note_{midi_note:03d}.wav")
    wavfile.write(filename, SAMPLE_RATE, wave)
    print(f"Generated {filename} ({freq:.2f} Hz)")
