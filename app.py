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

APP_VERSION = "CHIPTUNE-MUSICAL-003"
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
    '<div class="sub-title">Musical 8-Bit Re-Arrangement</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

이번 버전은 억지로 원곡 전체 음을 하나하나 따라가기보다,
원곡의 BPM/키/에너지 흐름을 분석해서 더 자연스러운 8비트 편곡처럼 만들게 바꿨어.
소리도 삑삑거리는 효과음 느낌을 줄이고, 부드러운 리드/베이스/아르페지오/칩 드럼으로 바꿨어.

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
# Helpers
# -------------------------
NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def safe_bpm(bpm):
    if bpm <= 0 or np.isnan(bpm):
        return 128

    while bpm < 85:
        bpm *= 2

    while bpm > 180:
        bpm /= 2

    return int(bpm)


def np_to_pydub(samples):
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)

    return AudioSegment(
        int_samples.tobytes(),
        frame_rate=SYNTH_SR,
        sample_width=2,
        channels=2
    )


def make_wave(freq, duration_sec, wave_type="soft_pulse", amp=0.25):
    length = max(1, int(SYNTH_SR * duration_sec))
    t = np.linspace(0, duration_sec, length, False)

    if wave_type == "soft_pulse":
        raw = np.sin(2 * np.pi * freq * t)
        wave = np.where(raw > 0.15, 1.0, -1.0)
        wave = np.tanh(wave * 1.2)

    elif wave_type == "triangle":
        wave = 2 * np.abs(2 * ((freq * t) % 1) - 1) - 1

    elif wave_type == "sine_chip":
        wave = np.sin(2 * np.pi * freq * t)
        wave = np.round(wave * 8) / 8

    elif wave_type == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))

    else:
        wave = np.sin(2 * np.pi * freq * t)

    attack = max(1, int(0.008 * SYNTH_SR))
    release = max(1, int(0.045 * SYNTH_SR))

    envelope = np.ones(length)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)

    wave = wave * envelope * amp
    return np.column_stack([wave, wave])


def add_wave(buffer, start_ms, duration_ms, midi, wave_type, amp):
    start = int(SYNTH_SR * start_ms / 1000)

    if start >= len(buffer):
        return

    freq = midi_to_freq(midi)
    duration_sec = max(0.04, duration_ms / 1000)

    wave = make_wave(
        freq,
        duration_sec,
        wave_type=wave_type,
        amp=amp
    )

    end = min(len(buffer), start + len(wave))

    if end > start:
        buffer[start:end] += wave[:end - start]


def add_noise(buffer, start_ms, duration_ms=60, amp=0.08):
    start = int(SYNTH_SR * start_ms / 1000)

    if start >= len(buffer):
        return

    length = int(SYNTH_SR * duration_ms / 1000)
    noise = np.random.uniform(-1, 1, length)
    env = np.linspace(1, 0, length) ** 1.6
    noise = noise * env * amp
    stereo = np.column_stack([noise, noise])

    end = min(len(buffer), start + len(stereo))

    if end > start:
        buffer[start:end] += stereo[:end - start]


def add_kick(buffer, start_ms, amp=0.16):
    start = int(SYNTH_SR * start_ms / 1000)

    if start >= len(buffer):
        return

    duration_ms = 95
    length = int(SYNTH_SR * duration_ms / 1000)

    freqs = np.linspace(105, 48, length)
    phase = 2 * np.pi * np.cumsum(freqs) / SYNTH_SR

    wave = np.sin(phase)
    wave = np.round(wave * 5) / 5

    env = np.linspace(1, 0, length) ** 2
    wave = wave * env * amp

    stereo = np.column_stack([wave, wave])

    end = min(len(buffer), start + len(stereo))

    if end > start:
        buffer[start:end] += stereo[:end - start]


# -------------------------
# Music theory helpers
# -------------------------
def get_scale_for_key(root, mode):
    if mode == "minor":
        intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]

    return [(root + i) % 12 for i in intervals]


def pc_to_midi(pc, low=60, high=84, previous=None):
    candidates = []

    for octave in range(0, 9):
        midi = pc + octave * 12

        if low <= midi <= high:
            candidates.append(midi)

    if not candidates:
        return low

    if previous is None:
        center = (low + high) // 2
        return min(candidates, key=lambda x: abs(x - center))

    return min(candidates, key=lambda x: abs(x - previous))


def nearest_scale_pc(pc, scale):
    best = pc
    best_dist = 99

    for scale_pc in scale:
        dist = min(
            abs(pc - scale_pc),
            12 - abs(pc - scale_pc)
        )

        if dist < best_dist:
            best = scale_pc
            best_dist = dist

    return best


def chord_from_root(root_pc, scale, mode):
    if mode == "minor":
        third = (root_pc + 3) % 12
    else:
        third = (root_pc + 4) % 12

    fifth = (root_pc + 7) % 12

    third = nearest_scale_pc(third, scale)
    fifth = nearest_scale_pc(fifth, scale)

    return [root_pc, third, fifth]


# -------------------------
# Analysis
# -------------------------
def analyze_song(file_path):
    if not LIBROSA_OK:
        return {
            "bpm": 128,
            "key_root": 0,
            "key_mode": "major",
            "sections": []
        }

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

    bpm = safe_bpm(bpm)

    hop_length = 512

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length
    )

    rms = librosa.feature.rms(
        y=y,
        hop_length=hop_length
    )[0]

    times = librosa.frames_to_time(
        np.arange(chroma.shape[1]),
        sr=sr,
        hop_length=hop_length
    ) * 1000

    key_root, key_mode = estimate_key(
        chroma
    )

    scale = get_scale_for_key(
        key_root,
        key_mode
    )

    beat_ms = int(60000 / bpm)
    bar_ms = beat_ms * 4

    total_ms = int(len(y) / sr * 1000)

    sections = []
    pos = 0
    prev_lead = None

    energy_floor = np.percentile(rms, 35)

    while pos < total_ms:
        end = min(total_ms, pos + bar_ms)

        mask = (times >= pos) & (times < end)

        if np.sum(mask) < 2:
            pos += bar_ms
            continue

        chroma_slice = chroma[:, mask]
        energy = float(np.mean(rms[mask]))

        if energy < energy_floor * 0.55:
            pos += bar_ms
            continue

        chroma_energy = np.mean(chroma_slice, axis=1)
        ranked = list(np.argsort(chroma_energy)[::-1])

        root_pc = nearest_scale_pc(int(ranked[0]), scale)

        chord = chord_from_root(
            root_pc,
            scale,
            key_mode
        )

        # lead는 가장 강한 음을 쓰되 키 안으로 보정
        lead_pc = nearest_scale_pc(
            int(ranked[0]),
            scale
        )

        lead_midi = pc_to_midi(
            lead_pc,
            low=64,
            high=88,
            previous=prev_lead
        )

        prev_lead = lead_midi

        bass_midi = pc_to_midi(
            root_pc,
            low=36,
            high=48
        )

        sections.append(
            {
                "start": int(pos),
                "duration": int(end - pos),
                "root_pc": root_pc,
                "chord": chord,
                "lead": lead_midi,
                "bass": bass_midi,
                "energy": energy
            }
        )

        pos += bar_ms

    return {
        "bpm": bpm,
        "key_root": key_root,
        "key_mode": key_mode,
        "sections": sections
    }


def estimate_key(chroma):
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

    return best_root, best_mode


# -------------------------
# Arrangement
# -------------------------
def make_progression_if_empty(total_ms, bpm, key_root, key_mode):
    scale = get_scale_for_key(key_root, key_mode)
    beat_ms = int(60000 / bpm)
    bar_ms = beat_ms * 4

    progression_degrees = [0, 5, 3, 4]
    sections = []

    pos = 0
    i = 0

    while pos < total_ms:
        root_pc = scale[progression_degrees[i % len(progression_degrees)] % len(scale)]
        chord = chord_from_root(root_pc, scale, key_mode)

        sections.append(
            {
                "start": pos,
                "duration": bar_ms,
                "root_pc": root_pc,
                "chord": chord,
                "lead": pc_to_midi(root_pc, low=67, high=84),
                "bass": pc_to_midi(root_pc, low=36, high=48),
                "energy": 1.0
            }
        )

        pos += bar_ms
        i += 1

    return sections


def synthesize_musical_chiptune(original_audio, analysis, style, strength, guide):
    total_ms = len(original_audio)
    bpm = analysis["bpm"]
    key_root = analysis["key_root"]
    key_mode = analysis["key_mode"]
    sections = analysis["sections"]

    if len(sections) < 2:
        sections = make_progression_if_empty(
            total_ms,
            bpm,
            key_root,
            key_mode
        )

    length_samples = int(SYNTH_SR * total_ms / 1000) + SYNTH_SR
    buffer = np.zeros((length_samples, 2), dtype=np.float32)

    if strength == "Light":
        lead_amp = 0.11
        chord_amp = 0.045
        bass_amp = 0.10
        arp_amp = 0.055
        drum_amp = 0.08

    elif strength == "Strong":
        lead_amp = 0.25
        chord_amp = 0.10
        bass_amp = 0.20
        arp_amp = 0.12
        drum_amp = 0.16

    else:
        lead_amp = 0.17
        chord_amp = 0.07
        bass_amp = 0.15
        arp_amp = 0.085
        drum_amp = 0.12

    if style == "Soft":
        lead_wave = "sine_chip"
        chord_wave = "triangle"
        bass_wave = "triangle"

    elif style == "Gameboy":
        lead_wave = "soft_pulse"
        chord_wave = "triangle"
        bass_wave = "triangle"

    elif style == "NES":
        lead_wave = "square"
        chord_wave = "soft_pulse"
        bass_wave = "triangle"

    else:
        lead_wave = "soft_pulse"
        chord_wave = "square"
        bass_wave = "triangle"

    beat_ms = int(60000 / bpm)
    step_ms = max(90, beat_ms // 2)

    # -------------------------
    # Chords and bass follow original harmony
    # -------------------------
    for section in sections:
        start = section["start"]
        duration = section["duration"]
        chord = section["chord"]
        bass = section["bass"]

        # pad/chord
        for pc in chord:
            midi = pc_to_midi(pc, low=60, high=76)
            add_wave(
                buffer,
                start,
                duration,
                midi,
                chord_wave,
                chord_amp
            )

        # bass pulse
        pos = start

        while pos < start + duration:
            add_wave(
                buffer,
                pos,
                int(beat_ms * 0.85),
                bass,
                bass_wave,
                bass_amp
            )

            pos += beat_ms

    # -------------------------
    # Lead melody: chord tones + motion, not random repeated sound
    # -------------------------
    phrase_index = 0

    for section in sections:
        start = section["start"]
        duration = section["duration"]
        chord = section["chord"]

        chord_midis = [
            pc_to_midi(pc, low=67, high=88)
            for pc in chord
        ]

        if style == "Boss":
            pattern = [0, 1, 2, 1, 2, 1, 0, 2]
        elif style == "Cute":
            pattern = [0, 1, 2, 1, 0, 2, 1, 2]
        else:
            pattern = [0, 1, 2, 1, 2, 0, 1, 2]

        pos = start
        note_i = 0

        while pos < start + duration:
            midi = chord_midis[
                pattern[(note_i + phrase_index) % len(pattern)] % len(chord_midis)
            ]

            if note_i % 8 in [3, 7]:
                midi += 12

            if midi > 88:
                midi -= 12

            add_wave(
                buffer,
                pos,
                int(step_ms * 0.72),
                midi,
                lead_wave,
                lead_amp
            )

            pos += step_ms
            note_i += 1

        phrase_index += 1

    # -------------------------
    # Arpeggio
    # -------------------------
    if strength != "Light":
        for section in sections:
            start = section["start"]
            duration = section["duration"]
            chord = section["chord"]

            chord_midis = [
                pc_to_midi(pc, low=72, high=91)
                for pc in chord
            ]

            pos = start
            arp_i = 0
            arp_step = max(80, beat_ms // 4)

            while pos < start + duration:
                midi = chord_midis[arp_i % len(chord_midis)]

                add_wave(
                    buffer,
                    pos,
                    int(arp_step * 0.55),
                    midi,
                    "soft_pulse",
                    arp_amp
                )

                pos += arp_step
                arp_i += 1

    # -------------------------
    # Drums
    # -------------------------
    beat_count = 0

    for pos in range(0, total_ms - 1000, beat_ms):
        if beat_count % 2 == 0:
            add_kick(
                buffer,
                pos,
                amp=drum_amp
            )

        if beat_count % 4 == 2:
            add_noise(
                buffer,
                pos,
                duration_ms=75,
                amp=drum_amp * 0.75
            )

        add_noise(
            buffer,
            pos + beat_ms // 2,
            duration_ms=30,
            amp=drum_amp * 0.42
        )

        if strength == "Strong":
            add_noise(
                buffer,
                pos + beat_ms // 4,
                duration_ms=25,
                amp=drum_amp * 0.28
            )

        beat_count += 1

    result = np_to_pydub(
        buffer[:int(SYNTH_SR * total_ms / 1000)]
    )

    # 원곡 가이드
    if guide != "None":
        if guide == "Very Low":
            guide_gain = -24
        elif guide == "Low":
            guide_gain = -18
        else:
            guide_gain = -14

        original = original_audio.high_pass_filter(80)
        original = original.low_pass_filter(9000)
        original = original + guide_gain
        original = original.set_frame_rate(SYNTH_SR).set_channels(2)

        result = result.overlay(
            original,
            position=0
        )

    result = normalize(result)
    result = result.fade_in(120)
    result = result.fade_out(900)

    key_name = f"{NOTE_NAMES[key_root]} {key_mode}"

    return result, key_name, bpm, len(sections)


# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

st.markdown('<div class="control-box">', unsafe_allow_html=True)

style = st.selectbox(
    "Chip Sound",
    [
        "Soft",
        "Gameboy",
        "NES",
        "Boss"
    ],
    index=1
)

strength = st.selectbox(
    "Arrangement Strength",
    [
        "Light",
        "Normal",
        "Strong"
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
    "이 버전은 원곡에서 화음 흐름을 뽑아서 그 코드 진행 위에 8비트 리드/베이스/아르페지오를 다시 만드는 방식이야."
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

    if st.button("Rebuild as Musical 8-Bit"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing BPM...",
            "Detecting harmony flow...",
            "Creating 8-bit chord progression...",
            "Creating melodic chip lead...",
            "Adding bass and drums...",
            "Finalizing 8-bit arrangement..."
        ]

        for i in range(101):
            time.sleep(0.008)
            progress.progress(i)

            if i < 10:
                status.write(steps[0])
            elif i < 24:
                status.write(steps[1])
            elif i < 42:
                status.write(steps[2])
            elif i < 58:
                status.write(steps[3])
            elif i < 75:
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

        analysis = analyze_song(
            temp_path
        )

        rebuilt, key_name, bpm, section_count = synthesize_musical_chiptune(
            original_audio,
            analysis,
            style,
            strength,
            guide
        )

        output_path = "pacoel_musical_8bit_arrangement.mp3"

        rebuilt.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.converted_bytes = f.read()

        st.session_state.convert_info = (
            f"{style} / {strength} / Key {key_name} / BPM {bpm} / Sections {section_count} / {APP_VERSION}"
        )

    if st.session_state.converted_bytes:
        st.success("Musical 8-Bit Arrangement Complete!")

        st.write(f"Info: {st.session_state.convert_info}")

        st.audio(
            st.session_state.converted_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download Musical 8-Bit Arrangement",
            st.session_state.converted_bytes,
            file_name="pacoel_musical_8bit_arrangement.mp3",
            mime="audio/mpeg",
        )