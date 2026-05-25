import streamlit as st
from pydub import AudioSegment
from pydub.effects import normalize
import numpy as np
import tempfile
import time
import os

try:
    import librosa
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Pacoel Wave Chiptune",
    page_icon="🎮",
    layout="wide"
)

APP_VERSION = "CHIPTUNE-PITCH-MATCH-002"

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    color: #111111 !important;
}

.stApp {
    background-color: #d9d9dd;
    color: #111111;
}

.main-title {
    font-size: 60px;
    font-weight: 900;
    color: #111111;
}

.sub-title {
    font-size: 18px;
    color: #333333;
    margin-bottom: 35px;
}

.info-box, .control-box {
    background: #cfcfd4;
    color: #111111;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    border: 1px solid #b8b8bf;
}

.stFileUploader {
    background-color: #cfcfd4;
    color: #111111;
    padding: 16px;
    border-radius: 18px;
    border: 1px solid #b8b8bf;
}

.stButton button, .stDownloadButton button {
    background-color: #111111;
    color: #ffffff;
    border-radius: 16px;
    height: 54px;
    width: 100%;
    font-size: 18px;
    border: none;
}

div[data-testid="stMarkdownContainer"] {
    color: #111111;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# Header
# -------------------------
st.markdown(
    '<div class="main-title">🎮 Pacoel Wave Chiptune</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Pitch-Matched 8-Bit Recomposer</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

이번 버전은 원곡 위에 효과음만 얹는 방식이 아니야.<br>
원곡의 피치와 키를 분석해서, 가능한 원곡 음에 맞게 8-bit 신스로 다시 연주하는 방식이야.

</div>
""", unsafe_allow_html=True)


# -------------------------
# Session state
# -------------------------
if "app_version" not in st.session_state:
    st.session_state.app_version = APP_VERSION

if st.session_state.app_version != APP_VERSION:
    st.session_state.clear()
    st.session_state.app_version = APP_VERSION

if "converted_bytes" not in st.session_state:
    st.session_state.converted_bytes = None

if "source_name" not in st.session_state:
    st.session_state.source_name = None

if "convert_info" not in st.session_state:
    st.session_state.convert_info = None


# -------------------------
# Constants
# -------------------------
SYNTH_SR = 44100

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


# -------------------------
# Utility
# -------------------------
def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def clamp_melody_midi(midi_note):
    while midi_note < 60:
        midi_note += 12

    while midi_note > 84:
        midi_note -= 12

    return int(midi_note)


def clamp_bass_midi(midi_note):
    while midi_note > 48:
        midi_note -= 12

    while midi_note < 36:
        midi_note += 12

    return int(midi_note)


def match_audio(layer, target):
    return layer.set_frame_rate(
        target.frame_rate
    ).set_channels(
        target.channels
    )


def nearest_scale_midi(midi_note, allowed_pitch_classes):
    """
    추출된 음을 곡의 키 안에 있는 가장 가까운 음으로 보정.
    """
    best = midi_note
    best_distance = 99

    for candidate in range(midi_note - 6, midi_note + 7):
        if candidate % 12 in allowed_pitch_classes:
            distance = abs(candidate - midi_note)

            if distance < best_distance:
                best = candidate
                best_distance = distance

    return int(best)


def quantize_time(value_ms, grid_ms):
    if grid_ms <= 0:
        return int(value_ms)

    return int(round(value_ms / grid_ms) * grid_ms)


# -------------------------
# Synth generators
# -------------------------
def make_wave(freq, duration_ms, wave_type="square", volume_db=-8):
    duration = max(0.03, duration_ms / 1000)

    t = np.linspace(
        0,
        duration,
        int(SYNTH_SR * duration),
        False
    )

    if wave_type == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))

    elif wave_type == "triangle":
        wave = 2 * np.abs(2 * ((freq * t) % 1) - 1) - 1

    elif wave_type == "pulse":
        raw = np.sin(2 * np.pi * freq * t)
        wave = np.where(raw > 0.45, 1.0, -1.0)

    else:
        wave = np.sin(2 * np.pi * freq * t)

    attack_len = max(1, int(0.006 * SYNTH_SR))
    release_len = max(1, int(0.035 * SYNTH_SR))

    envelope = np.ones_like(wave)

    envelope[:attack_len] = np.linspace(
        0,
        1,
        attack_len
    )

    envelope[-release_len:] = np.linspace(
        1,
        0,
        release_len
    )

    wave = wave * envelope * 0.32

    samples = (wave * 32767).astype(np.int16)

    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=SYNTH_SR,
        sample_width=2,
        channels=1
    )

    return seg + volume_db


def make_noise_hit(duration_ms=70, volume_db=-18):
    length = int(SYNTH_SR * duration_ms / 1000)

    noise = np.random.uniform(
        -1,
        1,
        length
    )

    envelope = np.linspace(
        1,
        0,
        length
    )

    noise = noise * envelope * 0.35

    samples = (noise * 32767).astype(np.int16)

    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=SYNTH_SR,
        sample_width=2,
        channels=1
    )

    return seg + volume_db


def make_kick(volume_db=-9):
    duration_ms = 95
    length = int(SYNTH_SR * duration_ms / 1000)

    t = np.linspace(
        0,
        duration_ms / 1000,
        length,
        False
    )

    freq_start = 120
    freq_end = 55

    freqs = np.linspace(
        freq_start,
        freq_end,
        length
    )

    phase = 2 * np.pi * np.cumsum(freqs) / SYNTH_SR

    wave = np.sign(np.sin(phase))

    envelope = np.linspace(
        1,
        0,
        length
    ) ** 2

    wave = wave * envelope * 0.45

    samples = (wave * 32767).astype(np.int16)

    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=SYNTH_SR,
        sample_width=2,
        channels=1
    )

    return seg + volume_db


# -------------------------
# Key detection
# -------------------------
def estimate_key(y, sr):
    """
    곡의 대략적인 키를 추정해서 멜로디 음을 그 키 안으로 보정.
    """
    try:
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=4096,
            hop_length=1024
        )

        chroma_mean = np.mean(chroma, axis=1)

        major_intervals = [0, 2, 4, 5, 7, 9, 11]
        minor_intervals = [0, 2, 3, 5, 7, 8, 10]

        best_score = -1
        best_root = 0
        best_mode = "major"
        best_scale = major_intervals

        for root in range(12):
            major_pcs = [(root + i) % 12 for i in major_intervals]
            minor_pcs = [(root + i) % 12 for i in minor_intervals]

            major_score = sum(chroma_mean[pc] for pc in major_pcs)
            minor_score = sum(chroma_mean[pc] for pc in minor_pcs)

            if major_score > best_score:
                best_score = major_score
                best_root = root
                best_mode = "major"
                best_scale = major_pcs

            if minor_score > best_score:
                best_score = minor_score
                best_root = root
                best_mode = "minor"
                best_scale = minor_pcs

        key_name = f"{NOTE_NAMES[best_root]} {best_mode}"

        return key_name, best_scale

    except Exception:
        return "C major", [0, 2, 4, 5, 7, 9, 11]


# -------------------------
# Analysis
# -------------------------
def analyze_audio(file_path, sensitivity):
    if not LIBROSA_OK:
        return 128, [], "C major", [0, 2, 4, 5, 7, 9, 11]

    y, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    try:
        bpm = float(tempo[0])
    except Exception:
        bpm = float(tempo)

    if bpm <= 0 or np.isnan(bpm):
        bpm = 128

    key_name, scale_pcs = estimate_key(
        y,
        sr
    )

    notes = extract_melody_notes(
        y,
        sr,
        int(bpm),
        scale_pcs,
        sensitivity
    )

    return int(bpm), notes, key_name, scale_pcs


def smooth_midi_sequence(midi_values, window=5):
    smoothed = []

    half = window // 2

    for i in range(len(midi_values)):
        nearby = []

        for j in range(i - half, i + half + 1):
            if 0 <= j < len(midi_values):
                if midi_values[j] is not None:
                    nearby.append(midi_values[j])

        if len(nearby) == 0:
            smoothed.append(None)
        else:
            smoothed.append(int(round(np.median(nearby))))

    return smoothed


def extract_melody_notes(y, sr, bpm, scale_pcs, sensitivity):
    """
    원곡에서 피치를 추출하고, 곡의 키에 맞는 음으로 보정해서 note list 생성.
    """
    hop_length = 512

    if sensitivity == "Stable":
        prob_threshold = 0.82
        min_note_ms = 160

    elif sensitivity == "Detailed":
        prob_threshold = 0.68
        min_note_ms = 90

    else:
        prob_threshold = 0.75
        min_note_ms = 120

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=2048,
            hop_length=hop_length
        )
    except Exception:
        return []

    times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sr,
        hop_length=hop_length
    )

    midi_values = []

    for i, hz in enumerate(f0):
        valid = (
            hz is not None
            and not np.isnan(hz)
            and voiced_flag[i]
            and voiced_probs[i] >= prob_threshold
        )

        if not valid:
            midi_values.append(None)
        else:
            midi = int(round(librosa.hz_to_midi(hz)))

            midi = nearest_scale_midi(
                midi,
                scale_pcs
            )

            midi = clamp_melody_midi(
                midi
            )

            midi_values.append(midi)

    midi_values = smooth_midi_sequence(
        midi_values,
        window=5
    )

    beat_ms = int(60000 / max(80, bpm))
    grid_ms = max(60, beat_ms // 4)

    raw_notes = []
    current_midi = None
    start_time = None

    for i, midi in enumerate(midi_values):
        t_ms = int(times[i] * 1000)

        if midi is None:
            if current_midi is not None:
                duration = t_ms - start_time

                if duration >= min_note_ms:
                    raw_notes.append(
                        {
                            "start": start_time,
                            "duration": duration,
                            "midi": current_midi
                        }
                    )

                current_midi = None
                start_time = None

            continue

        if current_midi is None:
            current_midi = midi
            start_time = t_ms

        elif abs(midi - current_midi) > 1:
            duration = t_ms - start_time

            if duration >= min_note_ms:
                raw_notes.append(
                    {
                        "start": start_time,
                        "duration": duration,
                        "midi": current_midi
                    }
                )

            current_midi = midi
            start_time = t_ms

    if current_midi is not None and start_time is not None:
        end_ms = int(times[-1] * 1000)
        duration = end_ms - start_time

        if duration >= min_note_ms:
            raw_notes.append(
                {
                    "start": start_time,
                    "duration": duration,
                    "midi": current_midi
                }
            )

    # 시간도 비트 그리드에 맞춤
    quantized = []

    for note in raw_notes:
        start = quantize_time(
            note["start"],
            grid_ms
        )

        duration = max(
            grid_ms,
            quantize_time(note["duration"], grid_ms)
        )

        quantized.append(
            {
                "start": start,
                "duration": duration,
                "midi": note["midi"]
            }
        )

    # 같은 음이 가까이 붙으면 합침
    cleaned = []

    for note in quantized:
        if not cleaned:
            cleaned.append(note)
            continue

        prev = cleaned[-1]
        prev_end = prev["start"] + prev["duration"]

        if (
            abs(prev["midi"] - note["midi"]) <= 1
            and note["start"] - prev_end <= grid_ms
        ):
            prev["duration"] = (
                note["start"] + note["duration"] - prev["start"]
            )
        else:
            cleaned.append(note)

    return cleaned


# -------------------------
# Fallback melody
# -------------------------
def get_scale_midi(mode, scale_pcs):
    base_notes = []

    for midi in range(60, 85):
        if midi % 12 in scale_pcs:
            base_notes.append(midi)

    if len(base_notes) < 5:
        base_notes = [72, 74, 76, 77, 79, 81, 84]

    if mode == "Cute":
        return [n for n in base_notes if n >= 67]

    if mode == "Dark":
        return [n - 3 for n in base_notes]

    if mode == "Boss":
        return [n - 5 for n in base_notes]

    return base_notes


def make_fallback_notes(total_ms, bpm, mode, scale_pcs):
    scale = get_scale_midi(
        mode,
        scale_pcs
    )

    beat_ms = int(60000 / max(80, bpm))

    notes = []

    pattern = [0, 2, 4, 2, 5, 4, 2, 0]

    pos = 0
    index = 0

    while pos < total_ms - beat_ms:
        midi = scale[
            pattern[index % len(pattern)] % len(scale)
        ]

        notes.append(
            {
                "start": pos,
                "duration": int(beat_ms * 0.8),
                "midi": midi
            }
        )

        pos += beat_ms
        index += 1

    return notes


# -------------------------
# Drums
# -------------------------
def add_chip_drums(audio, bpm, strength):
    output = audio

    beat_ms = int(60000 / max(80, bpm))

    if strength == "Light":
        kick_db = -17
        snare_db = -22
        hat_db = -28

    elif strength == "Strong":
        kick_db = -8
        snare_db = -13
        hat_db = -19

    else:
        kick_db = -11
        snare_db = -16
        hat_db = -23

    kick = make_kick(
        volume_db=kick_db
    )

    snare = make_noise_hit(
        duration_ms=80,
        volume_db=snare_db
    )

    hat = make_noise_hit(
        duration_ms=35,
        volume_db=hat_db
    )

    kick = match_audio(kick, output)
    snare = match_audio(snare, output)
    hat = match_audio(hat, output)

    beat_count = 0

    for pos in range(0, len(output) - 1000, beat_ms):
        if beat_count % 2 == 0:
            output = output.overlay(
                kick,
                position=pos
            )

        if beat_count % 4 == 2:
            output = output.overlay(
                snare,
                position=pos
            )

        output = output.overlay(
            hat,
            position=pos + beat_ms // 2
        )

        if strength == "Strong":
            output = output.overlay(
                hat - 4,
                position=pos + beat_ms // 4
            )

        beat_count += 1

    return output


# -------------------------
# Chiptune recomposition
# -------------------------
def synthesize_chiptune(
    original_audio,
    bpm,
    extracted_notes,
    key_name,
    scale_pcs,
    mode,
    intensity,
    guide_option
):
    total_ms = len(original_audio)

    if len(extracted_notes) < 8:
        extracted_notes = make_fallback_notes(
            total_ms,
            bpm,
            mode,
            scale_pcs
        )

    canvas = AudioSegment.silent(
        duration=total_ms,
        frame_rate=SYNTH_SR
    ).set_channels(2)

    if intensity == "Light":
        lead_db = -11
        harmony_db = -19
        bass_db = -15
        drum_strength = "Light"
        arp_db = -20

    elif intensity == "Strong":
        lead_db = -4
        harmony_db = -12
        bass_db = -8
        drum_strength = "Strong"
        arp_db = -13

    else:
        lead_db = -7
        harmony_db = -16
        bass_db = -11
        drum_strength = "Normal"
        arp_db = -16

    # -------------------------
    # Lead melody: 원곡 피치 기반
    # -------------------------
    for note in extracted_notes:
        start = note["start"]

        if start >= total_ms:
            continue

        duration = min(
            note["duration"],
            total_ms - start
        )

        midi = clamp_melody_midi(
            note["midi"]
        )

        freq = midi_to_freq(midi)

        lead = make_wave(
            freq,
            duration,
            wave_type="square",
            volume_db=lead_db
        )

        lead = lead.fade_out(25)
        lead = match_audio(lead, canvas)

        canvas = canvas.overlay(
            lead,
            position=start
        )

        # harmony는 같은 키 안에서 3도 위
        harmony_midi = midi + 4

        harmony_midi = nearest_scale_midi(
            harmony_midi,
            scale_pcs
        )

        harmony_midi = clamp_melody_midi(
            harmony_midi
        )

        harmony = make_wave(
            midi_to_freq(harmony_midi),
            duration,
            wave_type="pulse",
            volume_db=harmony_db
        )

        harmony = harmony.fade_out(25)
        harmony = match_audio(harmony, canvas)

        if intensity != "Light":
            canvas = canvas.overlay(
                harmony,
                position=start
            )

    # -------------------------
    # Bass: lead note 기반으로 음 맞춤
    # -------------------------
    beat_ms = int(60000 / max(80, bpm))
    note_index = 0

    for pos in range(0, total_ms - beat_ms, beat_ms):
        while (
            note_index + 1 < len(extracted_notes)
            and extracted_notes[note_index + 1]["start"] <= pos
        ):
            note_index += 1

        midi = extracted_notes[note_index]["midi"]
        bass_midi = clamp_bass_midi(midi)

        bass_midi = nearest_scale_midi(
            bass_midi,
            scale_pcs
        )

        freq = midi_to_freq(bass_midi)

        bass = make_wave(
            freq,
            int(beat_ms * 0.9),
            wave_type="triangle",
            volume_db=bass_db
        )

        bass = bass.fade_out(45)
        bass = match_audio(bass, canvas)

        canvas = canvas.overlay(
            bass,
            position=pos
        )

    # -------------------------
    # Arp: 추정 키 기반
    # -------------------------
    scale = get_scale_midi(
        mode,
        scale_pcs
    )

    arp_step = max(95, beat_ms // 2)
    arp_index = 0

    for pos in range(0, total_ms - arp_step, arp_step):
        if intensity == "Light" and arp_index % 2 == 1:
            arp_index += 1
            continue

        midi = scale[arp_index % len(scale)]

        if arp_index % 8 in [3, 7]:
            midi += 12

        midi = clamp_melody_midi(
            midi
        )

        freq = midi_to_freq(midi)

        arp = make_wave(
            freq,
            int(arp_step * 0.55),
            wave_type="pulse",
            volume_db=arp_db
        )

        arp = arp.fade_out(20)
        arp = match_audio(arp, canvas)

        canvas = canvas.overlay(
            arp,
            position=pos
        )

        arp_index += 1

    # -------------------------
    # Drums
    # -------------------------
    canvas = add_chip_drums(
        canvas,
        bpm,
        drum_strength
    )

    # -------------------------
    # Original guide
    # -------------------------
    if guide_option != "None":
        if guide_option == "Very Low":
            guide_db = -24
        elif guide_option == "Low":
            guide_db = -18
        else:
            guide_db = -14

        guide = original_audio
        guide = guide.high_pass_filter(80)
        guide = guide.low_pass_filter(9000)
        guide = guide + guide_db

        guide = match_audio(
            guide,
            canvas
        )

        canvas = canvas.overlay(
            guide,
            position=0
        )

    canvas = normalize(canvas)
    canvas = canvas.fade_in(120)
    canvas = canvas.fade_out(900)

    return canvas, len(extracted_notes), key_name


# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

st.markdown('<div class="control-box">', unsafe_allow_html=True)

mode = st.selectbox(
    "Chiptune Style",
    [
        "Arcade",
        "Cute",
        "Dark",
        "Boss"
    ],
    index=0
)

intensity = st.selectbox(
    "Recompose Strength",
    [
        "Light",
        "Normal",
        "Strong"
    ],
    index=1
)

sensitivity = st.selectbox(
    "Pitch Tracking",
    [
        "Stable",
        "Normal",
        "Detailed"
    ],
    index=1
)

guide_option = st.selectbox(
    "Original Song Guide",
    [
        "None",
        "Very Low",
        "Low",
        "Medium"
    ],
    index=1
)

st.write(
    "Stable = 음이 덜 튐 / Normal = 추천 / Detailed = 더 많은 음을 따라감"
)

if st.button("Reset Result"):
    st.session_state.converted_bytes = None
    st.session_state.convert_info = None
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


if uploaded:
    if st.session_state.source_name != uploaded.name:
        st.session_state.converted_bytes = None
        st.session_state.convert_info = None
        st.session_state.source_name = uploaded.name

    st.audio(uploaded)

    if st.button("Recompose Song as Pitch-Matched 8-Bit"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing key and BPM...",
            "Tracking melody pitch...",
            "Snapping notes to song key...",
            "Replaying melody with 8-bit synth...",
            "Adding tuned bass and arp...",
            "Finalizing chiptune version..."
        ]

        for i in range(101):
            time.sleep(0.008)
            progress.progress(i)

            if i < 10:
                status.write(steps[0])
            elif i < 24:
                status.write(steps[1])
            elif i < 40:
                status.write(steps[2])
            elif i < 55:
                status.write(steps[3])
            elif i < 70:
                status.write(steps[4])
            elif i < 88:
                status.write(steps[5])
            else:
                status.write(steps[6])

        file_ext = os.path.splitext(uploaded.name)[1]

        if file_ext.lower() not in [".mp3", ".wav"]:
            file_ext = ".mp3"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            uploaded.seek(0)
            tmp.write(uploaded.read())
            temp_path = tmp.name

        original_audio = AudioSegment.from_file(
            temp_path
        )

        if len(original_audio) < 5000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        bpm, notes, key_name, scale_pcs = analyze_audio(
            temp_path,
            sensitivity
        )

        chiptune, note_count, detected_key = synthesize_chiptune(
            original_audio,
            bpm,
            notes,
            key_name,
            scale_pcs,
            mode,
            intensity,
            guide_option
        )

        output_path = "pacoel_pitch_matched_8bit.mp3"

        chiptune.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.converted_bytes = f.read()

        st.session_state.convert_info = (
            f"{mode} / {intensity} / {sensitivity} / "
            f"BPM {bpm} / Key {detected_key} / Notes {note_count} / {APP_VERSION}"
        )

    if st.session_state.converted_bytes:
        st.success("Pitch-Matched 8-Bit Recomposition Complete!")

        st.write(f"Info: {st.session_state.convert_info}")

        st.audio(
            st.session_state.converted_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download Pitch-Matched 8-Bit Recomposition",
            st.session_state.converted_bytes,
            file_name="pacoel_pitch_matched_8bit.mp3",
            mime="audio/mpeg",
        )