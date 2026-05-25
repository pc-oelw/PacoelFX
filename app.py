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


st.set_page_config(
    page_title="Pacoel Wave",
    page_icon="🎮",
    layout="wide"
)

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

st.markdown(
    '<div class="main-title">🎮 Pacoel Wave</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Clean 8-Bit Remix Converter</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

This version does not simply destroy the audio quality.
It keeps the original song recognizable, adds clean 8-bit synth layers,
retro bass, arcade drums, and only a small amount of digital texture.

</div>
""", unsafe_allow_html=True)


# -------------------------
# Session state
# -------------------------
if "converted_bytes" not in st.session_state:
    st.session_state.converted_bytes = None

if "source_name" not in st.session_state:
    st.session_state.source_name = None

if "convert_info" not in st.session_state:
    st.session_state.convert_info = None


# -------------------------
# Basic helpers
# -------------------------
SAMPLE_RATE = 22050


def match_audio(layer, target):
    return layer.set_frame_rate(
        target.frame_rate
    ).set_channels(
        target.channels
    )


def analyze_bpm(file_path):
    if not LIBROSA_OK:
        return 120

    try:
        y, sr = librosa.load(
            file_path,
            sr=None,
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
            bpm = 120

        return int(bpm)

    except Exception:
        return 120


# -------------------------
# Synth generators
# -------------------------
def square_wave(freq, duration_ms, volume_db=-18):
    duration = duration_ms / 1000

    t = np.linspace(
        0,
        duration,
        int(SAMPLE_RATE * duration),
        False
    )

    wave = np.sign(
        np.sin(2 * np.pi * freq * t)
    )

    # 너무 거칠지 않게 살짝 부드럽게
    wave = wave * 0.25

    samples = (wave * 32767).astype(np.int16)

    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )

    return seg + volume_db


def triangle_wave(freq, duration_ms, volume_db=-20):
    duration = duration_ms / 1000

    t = np.linspace(
        0,
        duration,
        int(SAMPLE_RATE * duration),
        False
    )

    wave = 2 * np.abs(
        2 * ((freq * t) % 1) - 1
    ) - 1

    wave = wave * 0.22

    samples = (wave * 32767).astype(np.int16)

    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )

    return seg + volume_db


def noise_hit(duration_ms=70, volume_db=-22):
    samples = np.random.uniform(
        -1,
        1,
        int(SAMPLE_RATE * duration_ms / 1000)
    )

    envelope = np.linspace(
        1,
        0,
        len(samples)
    )

    samples = samples * envelope * 0.22

    samples = (samples * 32767).astype(np.int16)

    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )

    return seg + volume_db


# -------------------------
# Light retro texture
# -------------------------
def light_bit_texture(audio, amount=0.18):
    """
    음질을 박살내지 않고 살짝 디지털 질감만 추가.
    """
    base = audio.set_sample_width(2)

    samples = np.array(
        base.get_array_of_samples()
    ).astype(np.float32)

    if base.channels == 2:
        samples = samples.reshape((-1, 2))

    max_amp = 32767.0

    # 10-bit 정도라서 너무 깨지지 않음
    levels = 2 ** 10

    crushed = np.round(
        samples / max_amp * (levels / 2)
    ) / (levels / 2) * max_amp

    mixed = samples * (1 - amount) + crushed * amount

    mixed = np.clip(
        mixed,
        -32768,
        32767
    ).astype(np.int16)

    return AudioSegment(
        mixed.tobytes(),
        frame_rate=base.frame_rate,
        sample_width=2,
        channels=base.channels
    )


def retro_clean_eq(audio):
    # 먹먹함 방지: 저음 뭉침만 살짝 제거, 고음은 많이 남김
    result = audio.high_pass_filter(45)
    result = result.low_pass_filter(15000)
    return result


# -------------------------
# Music scale
# -------------------------
def get_scale(mode):
    if mode == "Cute":
        return [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 659.25]

    if mode == "Dark":
        return [246.94, 277.18, 311.13, 369.99, 415.30, 466.16, 554.37]

    if mode == "Boss":
        return [220.00, 246.94, 277.18, 329.63, 369.99, 415.30, 493.88]

    return [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 523.25]


# -------------------------
# Arrangement layers
# -------------------------
def add_chip_melody(audio, bpm, mode, strength):
    output = audio

    scale = get_scale(mode)

    beat_ms = int(60000 / max(80, bpm))

    if strength == "Light":
        step = beat_ms * 2
        volume = -24
    elif strength == "Strong":
        step = beat_ms // 2
        volume = -18
    else:
        step = beat_ms
        volume = -21

    pos = 0
    index = 0

    end_limit = len(output) - 600

    while pos < end_limit:
        freq = scale[index % len(scale)]

        if index % 8 in [3, 7]:
            freq *= 2

        note = square_wave(
            freq,
            int(step * 0.65),
            volume_db=volume
        )

        note = note.fade_out(45)
        note = match_audio(note, output)

        output = output.overlay(
            note,
            position=pos
        )

        pos += step
        index += 1

    return output


def add_chip_bass(audio, bpm, mode, strength):
    output = audio

    scale = get_scale(mode)

    root = scale[0] / 2
    fifth = scale[4] / 2

    beat_ms = int(60000 / max(80, bpm))

    if strength == "Light":
        volume = -22
    elif strength == "Strong":
        volume = -15
    else:
        volume = -18

    pos = 0
    count = 0

    end_limit = len(output) - beat_ms

    while pos < end_limit:
        freq = root if count % 2 == 0 else fifth

        bass = triangle_wave(
            freq,
            int(beat_ms * 0.85),
            volume_db=volume
        )

        bass = bass.fade_out(60)
        bass = match_audio(bass, output)

        output = output.overlay(
            bass,
            position=pos
        )

        pos += beat_ms
        count += 1

    return output


def add_arcade_drums(audio, bpm, strength):
    output = audio

    beat_ms = int(60000 / max(80, bpm))

    if strength == "Light":
        kick_vol = -23
        snare_vol = -26
        hat_vol = -30
    elif strength == "Strong":
        kick_vol = -16
        snare_vol = -20
        hat_vol = -25
    else:
        kick_vol = -19
        snare_vol = -23
        hat_vol = -28

    kick = square_wave(
        75,
        80,
        volume_db=kick_vol
    ).fade_out(60)

    snare = noise_hit(
        80,
        volume_db=snare_vol
    )

    hat = noise_hit(
        35,
        volume_db=hat_vol
    )

    kick = match_audio(kick, output)
    snare = match_audio(snare, output)
    hat = match_audio(hat, output)

    pos = 0
    beat_count = 0

    end_limit = len(output) - 900

    while pos < end_limit:
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

        pos += beat_ms
        beat_count += 1

    return output


# -------------------------
# Main remix function
# -------------------------
def make_clean_8bit_remix(audio, bpm, mode, strength):
    original = normalize(audio)

    # 원곡은 너무 깨지지 않게 유지
    base = original - 3
    base = retro_clean_eq(base)
    base = light_bit_texture(base, amount=0.16)

    total = len(base)

    intro_len = int(total * 0.28)
    build_len = int(total * 0.30)

    intro = base[:intro_len]
    build = base[intro_len:intro_len + build_len]
    drop = base[intro_len + build_len:]

    # 초반: 멜로디만 약하게
    intro = add_chip_melody(
        intro,
        bpm,
        mode,
        "Light"
    )

    # 중간: 베이스 + 약한 드럼
    build = add_chip_bass(
        build,
        bpm,
        mode,
        strength
    )

    build = add_arcade_drums(
        build,
        bpm,
        "Light" if strength == "Light" else "Normal"
    )

    build = build + 1

    # 후반: 가장 8-bit 편곡 느낌
    drop = add_chip_bass(
        drop,
        bpm,
        mode,
        "Strong" if strength != "Light" else "Normal"
    )

    drop = add_chip_melody(
        drop,
        bpm,
        mode,
        "Strong" if strength == "Strong" else "Normal"
    )

    drop = add_arcade_drums(
        drop,
        bpm,
        "Strong" if strength != "Light" else "Normal"
    )

    drop = drop + 2
    drop = drop.fade_out(1000)

    pause = AudioSegment.silent(
        duration=100
    )

    remix = intro.append(
        build,
        crossfade=200
    )

    remix = remix.append(
        pause,
        crossfade=5
    )

    remix = remix.append(
        drop,
        crossfade=120
    )

    remix = remix.fade_in(120)
    remix = remix.fade_out(900)

    return normalize(remix)


# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

st.markdown('<div class="control-box">', unsafe_allow_html=True)

mode = st.selectbox(
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
    "Remix Strength",
    [
        "Light",
        "Normal",
        "Strong"
    ],
    index=1
)

st.write(
    "Light = 원곡 중심 / Normal = 추천 / Strong = 8-bit 편곡감 강하게"
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    if st.session_state.source_name != uploaded.name:
        st.session_state.converted_bytes = None
        st.session_state.convert_info = None
        st.session_state.source_name = uploaded.name

    st.audio(uploaded)

    if st.button("Generate Clean 8-Bit Remix"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing BPM...",
            "Creating clean retro base...",
            "Adding 8-bit melody layer...",
            "Adding arcade bass and drums...",
            "Finalizing remix..."
        ]

        for i in range(101):
            time.sleep(0.008)
            progress.progress(i)

            if i < 15:
                status.write(steps[0])
            elif i < 30:
                status.write(steps[1])
            elif i < 50:
                status.write(steps[2])
            elif i < 68:
                status.write(steps[3])
            elif i < 88:
                status.write(steps[4])
            else:
                status.write(steps[5])

        file_ext = os.path.splitext(uploaded.name)[1]

        if file_ext.lower() not in [".mp3", ".wav"]:
            file_ext = ".mp3"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            uploaded.seek(0)
            tmp.write(uploaded.read())
            temp_path = tmp.name

        bpm = analyze_bpm(temp_path)

        audio = AudioSegment.from_file(temp_path)

        if len(audio) < 5000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        remix = make_clean_8bit_remix(
            audio,
            bpm,
            mode,
            strength
        )

        output_path = "pacoel_clean_8bit_remix.mp3"

        remix.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.converted_bytes = f.read()

        st.session_state.convert_info = f"{mode} / {strength} / BPM {bpm}"

    if st.session_state.converted_bytes:
        st.success("Clean 8-Bit Remix Complete!")

        st.write(f"Style: {st.session_state.convert_info}")

        st.audio(
            st.session_state.converted_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download Clean 8-Bit Remix",
            st.session_state.converted_bytes,
            file_name="pacoel_clean_8bit_remix.mp3",
            mime="audio/mpeg",
        )