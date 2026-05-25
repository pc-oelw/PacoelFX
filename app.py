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

APP_VERSION = "CHIPTUNE-RECOMPOSE-001"

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
    '<div class="sub-title">Melody-to-8-Bit Chiptune Recomposer</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

이번 버전은 원곡에 효과만 거는 방식이 아니야.<br>
원곡에서 멜로디를 분석한 뒤, 그 멜로디를 8-bit 사각파/삼각파 신스로 다시 연주해서
칩튠 편곡처럼 만드는 방식이야.

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


# -------------------------
# Utility
# -------------------------
def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def clamp_melody_midi(midi_note):
    # 8-bit 리드 멜로디가 잘 들리는 음역대로 이동
    while midi_note < 60:
        midi_note += 12

    while midi_note > 84:
        midi_note -= 12

    return midi_note


def clamp_bass_midi(midi_note):
    while midi_note > 48:
        midi_note -= 12

    while midi_note < 36:
        midi_note += 12

    return midi_note


def match_audio(layer, target):
    return layer.set_frame_rate(
        target.frame_rate
    ).set_channels(
        target.channels
    )


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

    # ADSR 느낌의 간단한 envelope
    attack_len = max(1, int(0.005 * SYNTH_SR))
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
    # 8-bit 킥: 낮은 사각파 + 빠른 decay
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
# Analysis
# -------------------------
def analyze_audio(file_path):
    if not LIBROSA_OK:
        return 128, []

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

    notes = extract_melody_notes(
        y,
        sr,
        int(bpm)
    )

    return int(bpm), notes


def extract_melody_notes(y, sr, bpm):
    """
    원곡에서 대표 멜로디 라인을 추출.
    완벽한 악보 추출은 아니지만,
    효과음 얹는 것보다 훨씬 '곡 자체를 8-bit로 다시 연주'하는 쪽에 가까움.
    """
    if not LIBROSA_OK:
        return []

    hop_length = 512

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

    for hz in f0:
        if hz is None or np.isnan(hz):
            midi_values.append(None)
        else:
            midi = int(round(librosa.hz_to_midi(hz)))
            midi_values.append(midi)

    notes = []
    current_midi = None
    start_time = None

    min_note_ms = 90

    for i, midi in enumerate(midi_values):
        t_ms = int(times[i] * 1000)

        if midi is None:
            if current_midi is not None:
                duration = t_ms - start_time

                if duration >= min_note_ms:
                    notes.append(
                        {
                            "start": start_time,
                            "duration": duration,
                            "midi": current_midi
                        }
                    )

                current_midi = None
                start_time = None

            continue

        midi = clamp_melody_midi(midi)

        if current_midi is None:
            current_midi = midi
            start_time = t_ms

        elif abs(midi - current_midi) > 1:
            duration = t_ms - start_time

            if duration >= min_note_ms:
                notes.append(
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
            notes.append(
                {
                    "start": start_time,
                    "duration": duration,
                    "midi": current_midi
                }
            )

    # 너무 촘촘하면 일부 정리
    cleaned = []

    for note in notes:
        if not cleaned:
            cleaned.append(note)
            continue

        prev = cleaned[-1]

        # 같은 음이 바로 이어지면 합침
        if (
            abs(prev["midi"] - note["midi"]) <= 1
            and note["start"] - (prev["start"] + prev["duration"]) < 120
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
def make_fallback_notes(total_ms, bpm, mode):
    scale = get_scale(mode)

    beat_ms = int(60000 / max(80, bpm))

    notes = []

    pattern = [0, 2, 4, 2, 5, 4, 2, 0]

    pos = 0
    index = 0

    while pos < total_ms - beat_ms:
        midi = scale[pattern[index % len(pattern)]]

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


def get_scale(mode):
    if mode == "Cute":
        return [72, 74, 76, 79, 81, 84, 88]

    if mode == "Dark":
        return [69, 71, 72, 76, 78, 81, 83]

    if mode == "Boss":
        return [67, 70, 72, 75, 79, 82, 84]

    return [72, 74, 76, 77, 79, 81, 84]


# -------------------------
# Chiptune recomposition
# -------------------------
def synthesize_chiptune(
    original_audio,
    bpm,
    extracted_notes,
    mode,
    intensity,
    guide_volume
):
    total_ms = len(original_audio)

    if len(extracted_notes) < 8:
        extracted_notes = make_fallback_notes(
            total_ms,
            bpm,
            mode
        )

    canvas = AudioSegment.silent(
        duration=total_ms,
        frame_rate=SYNTH_SR
    ).set_channels(2)

    if intensity == "Light":
        lead_db = -10
        bass_db = -14
        drum_strength = "Light"
        arp_db = -18

    elif intensity == "Strong":
        lead_db = -4
        bass_db = -8
        drum_strength = "Strong"
        arp_db = -12

    else:
        lead_db = -7
        bass_db = -11
        drum_strength = "Normal"
        arp_db = -15

    # -------------------------
    # Lead melody: 원곡에서 추출한 멜로디를 8-bit로 다시 연주
    # -------------------------
    for note in extracted_notes:
        start = note["start"]

        if start >= total_ms:
            continue

        duration = min(
            note["duration"],
            total_ms - start
        )

        midi = clamp_melody_midi(note["midi"])
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

    # -------------------------
    # Bass: 추출 멜로디를 기반으로 저음 재구성
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
    # Arp: 게임음악 느낌의 보조음
    # -------------------------
    scale = get_scale(mode)

    arp_step = max(95, beat_ms // 2)
    arp_index = 0

    for pos in range(0, total_ms - arp_step, arp_step):
        if intensity == "Light" and arp_index % 2 == 1:
            arp_index += 1
            continue

        midi = scale[arp_index % len(scale)]

        if arp_index % 8 in [3, 7]:
            midi += 12

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
    # Drums: 8-bit drum pattern
    # -------------------------
    canvas = add_chip_drums(
        canvas,
        bpm,
        drum_strength
    )

    # -------------------------
    # Optional original guide
    # 기본값은 거의 제거. 곡 파악용으로만 살짝 깔기.
    # -------------------------
    if guide_volume > 0:
        guide = original_audio

        guide = guide.high_pass_filter(70)
        guide = guide.low_pass_filter(10000)

        guide = guide - guide_volume
        guide = match_audio(guide, canvas)

        canvas = canvas.overlay(
            guide,
            position=0
        )

    canvas = normalize(canvas)
    canvas = canvas.fade_in(120)
    canvas = canvas.fade_out(900)

    return canvas


def add_chip_drums(audio, bpm, strength):
    output = audio

    beat_ms = int(60000 / max(80, bpm))

    if strength == "Light":
        kick_db = -15
        snare_db = -20
        hat_db = -26

    elif strength == "Strong":
        kick_db = -7
        snare_db = -12
        hat_db = -18

    else:
        kick_db = -10
        snare_db = -15
        hat_db = -22

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

guide_option = st.selectbox(
    "Original Song Guide",
    [
        "None",
        "Very Low",
        "Low"
    ],
    index=1
)

st.write(
    "이번 버전은 원곡 오디오에 효과를 거는 게 아니라, 멜로디를 뽑아서 8-bit 악기로 다시 연주해."
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

    if st.button("Recompose Song as 8-Bit"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing melody...",
            "Extracting main notes...",
            "Replaying melody with 8-bit synth...",
            "Adding chip bass...",
            "Adding arcade drums...",
            "Finalizing chiptune version..."
        ]

        for i in range(101):
            time.sleep(0.008)
            progress.progress(i)

            if i < 10:
                status.write(steps[0])
            elif i < 25:
                status.write(steps[1])
            elif i < 40:
                status.write(steps[2])
            elif i < 58:
                status.write(steps[3])
            elif i < 73:
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

        bpm, notes = analyze_audio(
            temp_path
        )

        if guide_option == "None":
            guide_volume = 999
        elif guide_option == "Very Low":
            guide_volume = 22
        else:
            guide_volume = 16

        chiptune = synthesize_chiptune(
            original_audio,
            bpm,
            notes,
            mode,
            intensity,
            guide_volume
        )

        output_path = "pacoel_recomposed_8bit.mp3"

        chiptune.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.converted_bytes = f.read()

        st.session_state.convert_info = (
            f"{mode} / {intensity} / BPM {bpm} / notes {len(notes)} / {APP_VERSION}"
        )

    if st.session_state.converted_bytes:
        st.success("8-Bit Recomposition Complete!")

        st.write(f"Info: {st.session_state.convert_info}")

        st.audio(
            st.session_state.converted_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download 8-Bit Recomposition",
            st.session_state.converted_bytes,
            file_name="pacoel_recomposed_8bit.mp3",
            mime="audio/mpeg",
        )