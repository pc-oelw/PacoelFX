import streamlit as st
from pydub import AudioSegment
import librosa
import numpy as np
import tempfile
import time
import os
import io

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Pacoel Wave",
    page_icon="🎮",
    layout="wide"
)

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

.info-box, .command-box {
    background: #cfcfd4;
    color: #111111;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    border: 1px solid #b8b8bf;
}

.stTextArea textarea {
    background-color: #e5e5e8 !important;
    color: #111111 !important;
    border-radius: 14px !important;
    border: 1px solid #9d9da5 !important;
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
    '<div class="main-title">🎮 Pacoel Wave</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">8-Bit Chiptune Remix Engine</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

Upload a song and Pacoel Wave will turn it into an 8-bit / chiptune-style remix.
It keeps the original song recognizable while adding square-wave synths,
retro drums, bitcrush texture, and arcade-style energy.

</div>
""", unsafe_allow_html=True)

# -------------------------
# Constants
# -------------------------
SAMPLE_RATE = 22050


# -------------------------
# Audio analysis
# -------------------------
def analyze_bpm(file_path):
    y, sr = librosa.load(
        file_path,
        sr=None,
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
        bpm = 140

    return int(bpm)


# -------------------------
# Command parser
# -------------------------
def parse_command(command):
    text = command.lower() if command else ""

    vibe = "arcade"
    intensity = "normal"
    target_bpm = None

    if "dark" in text or "어둡" in text:
        vibe = "dark"
    elif "cute" in text or "귀여" in text:
        vibe = "cute"
    elif "boss" in text or "보스" in text:
        vibe = "boss"
    elif "happy" in text or "밝" in text:
        vibe = "happy"

    if "hard" in text or "강" in text or "intense" in text:
        intensity = "hard"
    elif "soft" in text or "약" in text:
        intensity = "soft"

    words = text.replace("/", " ").replace(":", " ").split()

    for i, word in enumerate(words):
        if word == "bpm" and i + 1 < len(words):
            try:
                target_bpm = int(words[i + 1])
            except Exception:
                pass

    return vibe, intensity, target_bpm


# -------------------------
# Wave generators
# -------------------------
def square_wave(freq, duration_ms, volume_db=-12):
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

    samples = (wave * 0.35 * 32767).astype(np.int16)

    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )

    return seg + volume_db


def triangle_wave(freq, duration_ms, volume_db=-14):
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

    samples = (wave * 0.32 * 32767).astype(np.int16)

    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )

    return seg + volume_db


def noise_hit(duration_ms=80, volume_db=-15):
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

    samples = samples * envelope

    samples = (samples * 0.5 * 32767).astype(np.int16)

    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )

    return seg + volume_db


# -------------------------
# 8-bit / bitcrush effect
# -------------------------
def bitcrush(audio, bits=8, mix=0.45):
    base = audio.set_channels(1).set_sample_width(2)

    samples = np.array(
        base.get_array_of_samples()
    ).astype(np.float32)

    if len(samples) == 0:
        return audio

    max_amp = 32767
    levels = 2 ** bits

    crushed = np.round(
        samples / max_amp * levels
    ) / levels * max_amp

    crushed = np.clip(
        crushed,
        -32768,
        32767
    ).astype(np.int16)

    crushed_audio = AudioSegment(
        crushed.tobytes(),
        frame_rate=base.frame_rate,
        sample_width=2,
        channels=1
    )

    crushed_audio = crushed_audio.set_channels(
        audio.channels
    ).set_frame_rate(
        audio.frame_rate
    )

    crushed_audio = crushed_audio - int((1 - mix) * 12)

    original = audio - int(mix * 5)

    return original.overlay(
        crushed_audio
    )


# -------------------------
# Music helpers
# -------------------------
def get_scale(vibe):
    if vibe == "dark":
        return [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16]
    elif vibe == "cute":
        return [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 659.25]
    elif vibe == "boss":
        return [246.94, 277.18, 329.63, 369.99, 415.30, 493.88, 554.37]
    else:
        return [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 523.25]


def match_layer(layer, target):
    return layer.set_frame_rate(
        target.frame_rate
    ).set_channels(
        target.channels
    )


# -------------------------
# Chiptune layers
# -------------------------
def add_chip_melody(audio, bpm, vibe, intensity):
    output = audio

    scale = get_scale(vibe)

    beat_ms = int(60000 / max(80, bpm))

    if intensity == "hard":
        step = beat_ms // 2
        volume = -14
    elif intensity == "soft":
        step = beat_ms * 2
        volume = -20
    else:
        step = beat_ms
        volume = -17

    pos = 0
    index = 0

    while pos < len(output) - 500:
        freq = scale[index % len(scale)]

        if index % 4 == 3:
            freq *= 2

        note = square_wave(
            freq,
            int(step * 0.75),
            volume_db=volume
        )

        note = note.fade_out(40)
        note = match_layer(note, output)

        output = output.overlay(
            note,
            position=pos
        )

        pos += step
        index += 1

    return output


def add_chip_bass(audio, bpm, vibe, intensity):
    output = audio

    scale = get_scale(vibe)

    root = scale[0] / 2
    fifth = scale[4] / 2

    beat_ms = int(60000 / max(80, bpm))

    if intensity == "hard":
        volume = -9
    elif intensity == "soft":
        volume = -15
    else:
        volume = -12

    pos = 0
    count = 0

    while pos < len(output) - beat_ms:
        freq = root if count % 2 == 0 else fifth

        bass = square_wave(
            freq,
            int(beat_ms * 0.85),
            volume_db=volume
        )

        bass = bass.fade_out(60)
        bass = match_layer(bass, output)

        output = output.overlay(
            bass,
            position=pos
        )

        pos += beat_ms
        count += 1

    return output


def add_chip_drums(audio, bpm, intensity):
    output = audio

    beat_ms = int(60000 / max(80, bpm))

    if intensity == "hard":
        kick_vol = -5
        snare_vol = -10
        hat_vol = -18
    elif intensity == "soft":
        kick_vol = -11
        snare_vol = -16
        hat_vol = -24
    else:
        kick_vol = -8
        snare_vol = -13
        hat_vol = -21

    kick_sound = square_wave(
        70,
        90,
        volume_db=kick_vol
    ).fade_out(70)

    snare_sound = noise_hit(
        90,
        volume_db=snare_vol
    )

    hat_sound = noise_hit(
        35,
        volume_db=hat_vol
    )

    kick_sound = match_layer(kick_sound, output)
    snare_sound = match_layer(snare_sound, output)
    hat_sound = match_layer(hat_sound, output)

    pos = 0
    beat_count = 0

    end_limit = len(output) - 900

    while pos < end_limit:
        if beat_count % 2 == 0:
            output = output.overlay(
                kick_sound,
                position=pos
            )

        if beat_count % 4 == 2:
            output = output.overlay(
                snare_sound,
                position=pos
            )

        output = output.overlay(
            hat_sound,
            position=pos + beat_ms // 2
        )

        if intensity == "hard":
            output = output.overlay(
                hat_sound - 3,
                position=pos + beat_ms // 4
            )

        pos += beat_ms
        beat_count += 1

    return output


# -------------------------
# Section processing
# -------------------------
def process_intro(audio, bpm, vibe, intensity):
    section = bitcrush(
        audio,
        bits=8,
        mix=0.35
    )

    section = add_chip_melody(
        section,
        bpm,
        vibe,
        "soft"
    )

    section = section.fade_in(300)

    return section


def process_build(audio, bpm, vibe, intensity):
    section = bitcrush(
        audio,
        bits=7,
        mix=0.5
    )

    section = add_chip_bass(
        section,
        bpm,
        vibe,
        intensity
    )

    section = add_chip_drums(
        section,
        bpm,
        "soft" if intensity == "soft" else "normal"
    )

    section = section + 1

    return section


def process_drop(audio, bpm, vibe, intensity):
    section = bitcrush(
        audio,
        bits=6,
        mix=0.65
    )

    # 8-bit remix는 원곡을 너무 빠르게 망가뜨리지 않고,
    # 칩튠 악기와 드럼으로 편곡감을 줌
    section = add_chip_bass(
        section,
        bpm,
        vibe,
        "hard" if intensity != "soft" else "normal"
    )

    section = add_chip_melody(
        section,
        bpm,
        vibe,
        "hard" if intensity != "soft" else "normal"
    )

    section = add_chip_drums(
        section,
        bpm,
        "hard" if intensity != "soft" else "normal"
    )

    section = section + 2
    section = section.fade_out(1200)

    return section


# -------------------------
# Session state
# -------------------------
if "remix_bytes" not in st.session_state:
    st.session_state.remix_bytes = None

if "remix_bpm" not in st.session_state:
    st.session_state.remix_bpm = None

if "remix_source" not in st.session_state:
    st.session_state.remix_source = None

if "remix_info" not in st.session_state:
    st.session_state.remix_info = None


# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

st.markdown('<div class="command-box">', unsafe_allow_html=True)

user_command = st.text_area(
    "8-Bit Remix Command",
    value="""/style 8-bit chiptune arcade
/vibe rhythm game
/bpm auto
/intensity hard
/intro retro
/drop arcade boss
""",
    height=140
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    if st.session_state.remix_source != uploaded.name:
        st.session_state.remix_bytes = None
        st.session_state.remix_bpm = None
        st.session_state.remix_info = None
        st.session_state.remix_source = uploaded.name

    st.audio(uploaded)

    if st.button("Generate 8-Bit Remix"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing BPM...",
            "Reading remix command...",
            "Creating 8-bit intro...",
            "Creating chiptune build-up...",
            "Creating arcade drop...",
            "Finalizing remix..."
        ]

        for i in range(101):
            time.sleep(0.01)
            progress.progress(i)

            if i < 12:
                status.write(steps[0])
            elif i < 25:
                status.write(steps[1])
            elif i < 38:
                status.write(steps[2])
            elif i < 55:
                status.write(steps[3])
            elif i < 72:
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

        bpm = analyze_bpm(temp_path)

        vibe, intensity, target_bpm = parse_command(
            user_command
        )

        if target_bpm is not None:
            remix_bpm = target_bpm
        else:
            remix_bpm = bpm

        remix_bpm = max(
            90,
            min(remix_bpm,
                180)
        )

        audio = AudioSegment.from_file(temp_path)

        if len(audio) < 10000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        total = len(audio)

        intro_len = int(total * 0.30)
        build_len = int(total * 0.30)

        intro = audio[:intro_len]

        build = audio[
            intro_len:intro_len + build_len
        ]

        drop = audio[
            intro_len + build_len:
        ]

        intro = process_intro(
            intro,
            remix_bpm,
            vibe,
            intensity
        )

        build = process_build(
            build,
            remix_bpm,
            vibe,
            intensity
        )

        pause = AudioSegment.silent(
            duration=120
        )

        drop = process_drop(
            drop,
            remix_bpm,
            vibe,
            intensity
        )

        remix = intro.append(
            build,
            crossfade=250
        )

        remix = remix.append(
            pause,
            crossfade=5
        )

        remix = remix.append(
            drop,
            crossfade=120
        )

        remix = remix.fade_out(1000)

        output_path = "pacoel_8bit_remix.mp3"

        remix.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.remix_bytes = f.read()

        st.session_state.remix_bpm = remix_bpm
        st.session_state.remix_info = f"8-bit {vibe} / {intensity}"

    if st.session_state.remix_bytes:
        st.write(f"Detected / Remix BPM: {st.session_state.remix_bpm}")

        if st.session_state.remix_info:
            st.write(f"Style: {st.session_state.remix_info}")

        st.success("8-Bit Remix Complete!")

        st.audio(
            st.session_state.remix_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download 8-Bit Remix",
            st.session_state.remix_bytes,
            file_name="pacoel_8bit_remix.mp3",
            mime="audio/mpeg",
        )