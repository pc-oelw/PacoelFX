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

APP_VERSION = "CHROMA-CHIPTUNE-REBUILD-001"
SYNTH_SR = 44100


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
    '<div class="sub-title">Full Song 8-Bit Rebuilder</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

이번 버전은 같은 멜로디를 반복하는 방식이 아니라,
원곡 전체의 화음과 음 흐름을 분석해서 시간별로 8비트 리드, 코드, 베이스, 드럼을 다시 만드는 방식이야.
원곡을 효과음처럼 덮는 게 아니라, 곡 구조를 따라가게 만드는 버전이야.

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
# Basic helpers
# -------------------------
NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def pc_to_midi_near(pc, previous=None, low=60, high=84):
    candidates = []

    for octave in range(0, 10):
        midi = pc + octave * 12

        if low <= midi <= high:
            candidates.append(midi)

    if not candidates:
        return 72 + pc

    if previous is None:
        center = (low + high) // 2
        return min(candidates, key=lambda x: abs(x - center))

    return min(candidates, key=lambda x: abs(x - previous))


def pydub_to_np(audio):
    audio = audio.set_frame_rate(SYNTH_SR)
    audio = audio.set_channels(2)
    audio = audio.set_sample_width(2)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples.reshape((-1, 2))

    return samples / 32768.0


def np_to_pydub(samples):
    samples = np.clip(samples, -1.0, 1.0)

    int_samples = (samples * 32767).astype(np.int16)

    return AudioSegment(
        int_samples.tobytes(),
        frame_rate=SYNTH_SR,
        sample_width=2,
        channels=2
    )


def make_wave_array(freq, duration_sec, wave_type="square", amp=0.25):
    length = max(1, int(SYNTH_SR * duration_sec))

    t = np.linspace(0, duration_sec, length, False)

    if wave_type == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))

    elif wave_type == "pulse":
        raw = np.sin(2 * np.pi * freq * t)
        wave = np.where(raw > 0.35, 1.0, -1.0)

    elif wave_type == "triangle":
        wave = 2 * np.abs(2 * ((freq * t) % 1) - 1) - 1

    else:
        wave = np.sin(2 * np.pi * freq * t)

    attack = max(1, int(0.006 * SYNTH_SR))
    release = max(1, int(0.035 * SYNTH_SR))

    envelope = np.ones(length)

    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)

    wave = wave * envelope * amp

    stereo = np.column_stack([wave, wave])

    return stereo


def add_wave(buffer, start_ms, duration_ms, freq, wave_type, amp):
    start = int(SYNTH_SR * start_ms / 1000)
    duration_sec = max(0.03, duration_ms / 1000)

    wave = make_wave_array(
        freq,
        duration_sec,
        wave_type=wave_type,
        amp=amp
    )

    end = min(len(buffer), start + len(wave))

    if start >= len(buffer) or end <= start:
        return

    buffer[start:end] += wave[:end - start]


def add_noise_hit(buffer, start_ms, duration_ms=55, amp=0.12):
    start = int(SYNTH_SR * start_ms / 1000)
    length = int(SYNTH_SR * duration_ms / 1000)

    if start >= len(buffer):
        return

    noise = np.random.uniform(-1, 1, length)
    env = np.linspace(1, 0, length) ** 1.5

    noise = noise * env * amp
    stereo = np.column_stack([noise, noise])

    end = min(len(buffer), start + len(stereo))

    buffer[start:end] += stereo[:end - start]


def add_kick(buffer, start_ms, amp=0.22):
    start = int(SYNTH_SR * start_ms / 1000)

    duration_ms = 100
    length = int(SYNTH_SR * duration_ms / 1000)

    if start >= len(buffer):
        return

    freqs = np.linspace(125, 55, length)
    phase = 2 * np.pi * np.cumsum(freqs) / SYNTH_SR

    wave = np.sign(np.sin(phase))
    env = np.linspace(1, 0, length) ** 2

    wave = wave * env * amp
    stereo = np.column_stack([wave, wave])

    end = min(len(buffer), start + len(stereo))

    buffer[start:end] += stereo[:end - start]


# -------------------------
# Analysis
# -------------------------
def estimate_key_from_chroma(chroma):
    if chroma is None or chroma.shape[1] == 0:
        return "C major"

    chroma_mean = np.mean(chroma, axis=1)

    major = [0, 2, 4, 5, 7, 9, 11]
    minor = [0, 2, 3, 5, 7, 8, 10]

    best_score = -1
    best_root = 0
    best_mode = "major"

    for root in range(12):
        major_score = sum(chroma_mean[(root + x) % 12] for x in major)
        minor_score = sum(chroma_mean[(root + x) % 12] for x in minor)

        if major_score > best_score:
            best_score = major_score
            best_root = root
            best_mode = "major"

        if minor_score > best_score:
            best_score = minor_score
            best_root = root
            best_mode = "minor"

    return f"{NOTE_NAMES[best_root]} {best_mode}"


def get_top_pitch_classes(chroma_slice, previous_pc=None):
    energy = np.mean(chroma_slice, axis=1)

    if np.max(energy) <= 0:
        return [0, 4, 7]

    ranked = list(np.argsort(energy)[::-1])

    chosen = []

    for pc in ranked:
        if len(chosen) >= 3:
            break

        if energy[pc] < np.max(energy) * 0.38:
            continue

        chosen.append(int(pc))

    if len(chosen) == 0:
        if previous_pc is not None:
            chosen = [previous_pc]
        else:
            chosen = [0]

    while len(chosen) < 3:
        chosen.append(chosen[0])

    return chosen[:3]


def analyze_song(file_path, density):
    if not LIBROSA_OK:
        return 128, [], "Unknown"

    y, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    tempo, beats = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    try:
        bpm = float(tempo[0])
    except Exception:
        bpm = float(tempo)

    if bpm <= 0 or np.isnan(bpm):
        bpm = 128

    bpm = int(bpm)

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=512
    )

    rms = librosa.feature.rms(
        y=y,
        hop_length=512
    )[0]

    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]),
        sr=sr,
        hop_length=512
    ) * 1000

    key_name = estimate_key_from_chroma(
        chroma
    )

    beat_ms = int(60000 / max(80, bpm))

    if density == "Simple":
        step_ms = beat_ms
    elif density == "Detailed":
        step_ms = max(90, beat_ms // 4)
    else:
        step_ms = max(120, beat_ms // 2)

    total_ms = int(len(y) / sr * 1000)

    events = []

    previous_pc = None
    previous_lead_midi = None

    global_rms = np.percentile(rms, 40)

    pos = 0

    while pos < total_ms:
        end_pos = min(total_ms, pos + step_ms)

        mask = (frame_times >= pos) & (frame_times < end_pos)

        if np.sum(mask) == 0:
            pos += step_ms
            continue

        section_rms = np.mean(rms[mask])

        if section_rms < global_rms * 0.55:
            pos += step_ms
            continue

        chroma_slice = chroma[:, mask]

        pcs = get_top_pitch_classes(
            chroma_slice,
            previous_pc
        )

        lead_pc = pcs[0]
        chord_pc_1 = pcs[1]
        chord_pc_2 = pcs[2]

        lead_midi = pc_to_midi_near(
            lead_pc,
            previous_lead_midi,
            low=64,
            high=88
        )

        chord_midi_1 = pc_to_midi_near(
            chord_pc_1,
            lead_midi,
            low=60,
            high=84
        )

        chord_midi_2 = pc_to_midi_near(
            chord_pc_2,
            lead_midi,
            low=60,
            high=84
        )

        bass_midi = pc_to_midi_near(
            lead_pc,
            None,
            low=36,
            high=48
        )

        events.append(
            {
                "start": int(pos),
                "duration": int(step_ms * 0.85),
                "lead": int(lead_midi),
                "chord1": int(chord_midi_1),
                "chord2": int(chord_midi_2),
                "bass": int(bass_midi),
                "energy": float(section_rms)
            }
        )

        previous_pc = lead_pc
        previous_lead_midi = lead_midi

        pos += step_ms

    # 같은 음이 너무 길게 반복되면 리듬감 있게 살짝 나눔
    cleaned = []

    for event in events:
        if not cleaned:
            cleaned.append(event)
            continue

        prev = cleaned[-1]
        prev_end = prev["start"] + prev["duration"]

        if (
            event["lead"] == prev["lead"]
            and event["start"] - prev_end < step_ms * 0.35
            and prev["duration"] < step_ms * 2
        ):
            prev["duration"] = event["start"] + event["duration"] - prev["start"]
        else:
            cleaned.append(event)

    return bpm, cleaned, key_name


# -------------------------
# Chiptune synthesis
# -------------------------
def synthesize_chiptune(original_audio, bpm, events, key_name, style, strength, guide):
    total_ms = len(original_audio)

    if len(events) < 8:
        beat_ms = int(60000 / max(80, bpm))
        events = []

        pattern = [72, 76, 79, 76, 81, 79, 76, 72]

        pos = 0
        i = 0

        while pos < total_ms:
            midi = pattern[i % len(pattern)]

            events.append(
                {
                    "start": pos,
                    "duration": int(beat_ms * 0.85),
                    "lead": midi,
                    "chord1": midi + 4,
                    "chord2": midi + 7,
                    "bass": midi - 24,
                    "energy": 1.0
                }
            )

            pos += beat_ms
            i += 1

    length_samples = int(SYNTH_SR * total_ms / 1000) + SYNTH_SR
    buffer = np.zeros((length_samples, 2), dtype=np.float32)

    if strength == "Light":
        lead_amp = 0.16
        chord_amp = 0.055
        bass_amp = 0.12
        drum_amp = 0.10
        guide_db = -18

    elif strength == "Strong":
        lead_amp = 0.34
        chord_amp = 0.12
        bass_amp = 0.24
        drum_amp = 0.22
        guide_db = -26

    else:
        lead_amp = 0.24
        chord_amp = 0.08
        bass_amp = 0.18
        drum_amp = 0.16
        guide_db = -22

    if style == "Cute":
        lead_wave = "pulse"
        chord_wave = "triangle"

    elif style == "Dark":
        lead_wave = "square"
        chord_wave = "pulse"

    elif style == "Boss":
        lead_wave = "square"
        chord_wave = "square"

    else:
        lead_wave = "square"
        chord_wave = "triangle"

    # 원곡의 시간별 음을 그대로 따라가며 8비트 악기로 재구성
    for event in events:
        start = event["start"]
        duration = event["duration"]

        lead_freq = midi_to_freq(event["lead"])
        chord_freq_1 = midi_to_freq(event["chord1"])
        chord_freq_2 = midi_to_freq(event["chord2"])
        bass_freq = midi_to_freq(event["bass"])

        add_wave(
            buffer,
            start,
            duration,
            lead_freq,
            lead_wave,
            lead_amp
        )

        if strength != "Light":
            add_wave(
                buffer,
                start,
                duration,
                chord_freq_1,
                chord_wave,
                chord_amp
            )

            add_wave(
                buffer,
                start,
                duration,
                chord_freq_2,
                chord_wave,
                chord_amp * 0.8
            )

        # 베이스는 매 이벤트마다 말고, 일정 간격 느낌으로
        if start % int(60000 / max(80, bpm)) < 80:
            add_wave(
                buffer,
                start,
                duration,
                bass_freq,
                "triangle",
                bass_amp
            )

    # 칩튠 드럼
    beat_ms = int(60000 / max(80, bpm))

    beat_count = 0

    for pos in range(0, total_ms - 800, beat_ms):
        if beat_count % 2 == 0:
            add_kick(
                buffer,
                pos,
                amp=drum_amp
            )

        if beat_count % 4 == 2:
            add_noise_hit(
                buffer,
                pos,
                duration_ms=75,
                amp=drum_amp * 0.75
            )

        add_noise_hit(
            buffer,
            pos + beat_ms // 2,
            duration_ms=30,
            amp=drum_amp * 0.45
        )

        if strength == "Strong":
            add_noise_hit(
                buffer,
                pos + beat_ms // 4,
                duration_ms=25,
                amp=drum_amp * 0.32
            )

        beat_count += 1

    chiptune = np_to_pydub(buffer[:int(SYNTH_SR * total_ms / 1000)])
    chiptune = normalize(chiptune)

    # 원곡 가이드는 아주 작게만 섞어서 "원곡 느낌" 확인용
    if guide != "None":
        if guide == "Very Low":
            guide_gain = guide_db
        elif guide == "Low":
            guide_gain = guide_db + 4
        else:
            guide_gain = guide_db + 8

        original = original_audio
        original = original.high_pass_filter(90)
        original = original.low_pass_filter(8500)
        original = original + guide_gain
        original = original.set_frame_rate(SYNTH_SR).set_channels(2)

        chiptune = chiptune.overlay(
            original,
            position=0
        )

    chiptune = normalize(chiptune)
    chiptune = chiptune.fade_in(120)
    chiptune = chiptune.fade_out(900)

    return chiptune, len(events), key_name


# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

st.markdown('<div class="control-box">', unsafe_allow_html=True)

style = st.selectbox(
    "8-Bit Style",
    [
        "Arcade",
        "Cute",
        "Dark",
        "Boss"
    ],
    index=0
)

strength = st.selectbox(
    "Rebuild Strength",
    [
        "Light",
        "Normal",
        "Strong"
    ],
    index=1
)

density = st.selectbox(
    "Note Detail",
    [
        "Simple",
        "Normal",
        "Detailed"
    ],
    index=1
)

guide = st.selectbox(
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
    "Detailed = 원곡 음 변화 더 많이 따라감 / Simple = 더 단순한 게임음악 느낌"
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

    if st.button("Rebuild Whole Song as 8-Bit"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing whole song harmony...",
            "Extracting time-based notes...",
            "Rebuilding lead as 8-bit...",
            "Rebuilding chords and bass...",
            "Adding chiptune drums...",
            "Finalizing 8-bit song..."
        ]

        for i in range(101):
            time.sleep(0.008)
            progress.progress(i)

            if i < 10:
                status.write(steps[0])
            elif i < 25:
                status.write(steps[1])
            elif i < 42:
                status.write(steps[2])
            elif i < 60:
                status.write(steps[3])
            elif i < 76:
                status.write(steps[4])
            elif i < 90:
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

        bpm, events, key_name = analyze_song(
            temp_path,
            density
        )

        rebuilt, event_count, detected_key = synthesize_chiptune(
            original_audio,
            bpm,
            events,
            key_name,
            style,
            strength,
            guide
        )

        output_path = "pacoel_full_8bit_rebuild.mp3"

        rebuilt.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.converted_bytes = f.read()

        st.session_state.convert_info = (
            f"{style} / {strength} / {density} / "
            f"BPM {bpm} / Key {detected_key} / Events {event_count} / {APP_VERSION}"
        )

    if st.session_state.converted_bytes:
        st.success("Full 8-Bit Rebuild Complete!")

        st.write(f"Info: {st.session_state.convert_info}")

        st.audio(
            st.session_state.converted_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download Full 8-Bit Rebuild",
            st.session_state.converted_bytes,
            file_name="pacoel_full_8bit_rebuild.mp3",
            mime="audio/mpeg",
        )