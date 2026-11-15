import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

SR = 44100
DURATION = 10.0
N = int(SR * DURATION)


def lowpass(data, cutoff, sr=SR, order=2):
    nyq = 0.5 * sr
    b, a = butter(order, cutoff / nyq, btype="low")
    return lfilter(b, a, data)


def bandpass(data, low, high, sr=SR, order=2):
    nyq = 0.5 * sr
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, data)


def synth_guitar_note(freq, dur, vel=0.5):
    n = int(SR * dur)
    t = np.arange(n) / SR
    # Warm plucked acoustic guitar harmonic series
    sig = (
        1.00 * np.sin(2 * np.pi * freq * t)
        + 0.45 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-3.5 * t)
        + 0.22 * np.sin(2 * np.pi * 3 * freq * t) * np.exp(-5.0 * t)
        + 0.10 * np.sin(2 * np.pi * 4 * freq * t) * np.exp(-7.0 * t)
    )
    env = (1 - np.exp(-t / 0.008)) * np.exp(-t / 1.4)
    return lowpass(sig * env * vel, 2800)


def synth_piano_note(freq, dur, vel=0.35):
    n = int(SR * dur)
    t = np.arange(n) / SR
    sig = (
        1.0 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-2.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3 * freq * t) * np.exp(-4.0 * t)
    )
    env = (1 - np.exp(-t / 0.015)) * np.exp(-t / 2.2)
    return lowpass(sig * env * vel, 2200)


def synth_strings_pad(freqs, dur, vel=0.12):
    n = int(SR * dur)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for f in freqs:
        # Slight detune for warm ensemble
        sig += 0.5 * np.sin(2 * np.pi * f * t)
        sig += 0.25 * np.sin(2 * np.pi * (f * 1.002) * t)
        sig += 0.25 * np.sin(2 * np.pi * (f * 0.998) * t)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7
    return lowpass(sig * env * vel, 1400)


def add_at(buf, sig, start_sec):
    idx = int(start_sec * SR)
    end = min(len(buf), idx + len(sig))
    if end > idx:
        buf[idx:end] += sig[: end - idx]


def main():
    music = np.zeros(N, dtype=np.float64)

    # Gentle nostalgic progression: Dmaj7 -> Bm7 -> Gmaj7 -> A6 (slow tempo)
    # Guitar arpeggio
    guitar_notes = [
        (0.0, 146.83, 2.2), (0.45, 220.00, 1.8), (0.90, 277.18, 1.8), (1.35, 369.99, 1.8),
        (2.4, 123.47, 2.2), (2.85, 185.00, 1.8), (3.30, 246.94, 1.8), (3.75, 293.66, 1.8),
        (4.8, 98.00, 2.2),  (5.25, 196.00, 1.8), (5.70, 246.94, 1.8), (6.15, 369.99, 1.8),
        (7.2, 110.00, 2.5), (7.65, 220.00, 2.0), (8.10, 277.18, 2.0), (8.55, 329.63, 2.0),
    ]
    for t0, f, d in guitar_notes:
        add_at(music, synth_guitar_note(f, d, vel=0.22), t0)

    # Soft piano accents
    piano_notes = [
        (0.0, 587.33, 2.5), (2.4, 493.88, 2.5), (4.8, 392.00, 2.5), (7.2, 440.00, 2.8)
    ]
    for t0, f, d in piano_notes:
        add_at(music, synth_piano_note(f, d, vel=0.14), t0)

    # Quiet strings pad
    add_at(music, synth_strings_pad([146.83, 220.00, 277.18], 4.8, vel=0.06), 0.0)
    add_at(music, synth_strings_pad([196.00, 246.94, 329.63], 5.2, vel=0.06), 4.8)

    # Soft night crickets (subtle high-frequency chirps)
    rng = np.random.default_rng(42)
    t_all = np.arange(N) / SR
    cricket_carrier = np.sin(2 * np.pi * 4200 * t_all)
    # Chirp envelope every ~1.2s
    chirp_env = np.zeros(N)
    for ct in [0.6, 1.9, 3.2, 5.1, 6.8, 8.3]:
        for pulse in range(3):
            p0 = ct + pulse * 0.06
            i0 = int(p0 * SR)
            i1 = min(N, i0 + int(0.035 * SR))
            if i1 > i0:
                w = np.hanning(i1 - i0)
                chirp_env[i0:i1] += w
    crickets = cricket_carrier * chirp_env * 0.015

    # Star-chime on the streak at t = 4.2s
    chime_t = np.arange(int(1.8 * SR)) / SR
    chime = (
        0.6 * np.sin(2 * np.pi * 1174.66 * chime_t)
        + 0.4 * np.sin(2 * np.pi * 1760.00 * chime_t)
    ) * np.exp(-chime_t / 0.45) * 0.16
    sfx = np.zeros(N)
    add_at(sfx, chime, 4.2)

    # Faint clock tick at the end (t = 8.5, 9.1, 9.7)
    for tt in [8.5, 9.1, 9.7]:
        tick_n = int(0.03 * SR)
        tick_t = np.arange(tick_n) / SR
        tick = rng.normal(0, 1, tick_n) * np.exp(-tick_t / 0.005)
        tick = bandpass(tick, 1200, 3500) * 0.12
        add_at(sfx, tick, tt)

    # Load voiceover
    vo_sr, vo_data = wavfile.read("output/voiceover.wav")
    if vo_data.dtype == np.int16:
        vo_data = vo_data.astype(np.float64) / 32768.0
    if vo_data.ndim > 1:
        vo_data = vo_data.mean(axis=1)

    vo_track = np.zeros(N, dtype=np.float64)
    add_at(vo_track, vo_data * 0.92, 0.35)

    # Mix: keep voice dominant
    mix = vo_track + music * 0.55 + crickets + sfx
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    out_int16 = (np.clip(mix, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write("output/final_audio.wav", SR, out_int16)
    print("Wrote output/final_audio.wav successfully.")


if __name__ == "__main__":
    main()
