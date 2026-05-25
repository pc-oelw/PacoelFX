import streamlit as st
from pydub import AudioSegment
from pydub.effects import normalize
import numpy as np
import tempfile
import time
import os

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

.info-box {
    background: #cfcfd4;
    color: #111111;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    border: 1px solid #b8b8bf;
}

.control-box {
    background: #cfcfd4;
    color: #111111;
    padding: 20px;
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

.stButton button {
    background-color: #111111;
    color: #ffffff;
    border-radius: 16px;
    height: 54px;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stDownloadButton button {
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
    '<div class="sub-title">Full 8-Bit Audio Converter</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

Upload a song and Pacoel Wave will convert the whole track into a retro 8-bit style.
This version does not use AI and does not add random new melodies.
It transforms the entire audio using bitcrush, sample-rate reduction, and retro compression.

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
# Audio processing helpers
# -------------------------
def audiosegment_to_numpy(audio):
    audio = audio.set_sample_width(2)

    samples = np.array(
        audio.get_array_of_samples()
    ).astype(np.float32)

    if audio.channels == 2:
        samples = samples.reshape((-1, 2))

    return samples, audio.frame_rate, audio.channels


def numpy_to_audiosegment(samples, frame_rate, channels):
    samples = np.clip(
        samples,
        -32768,
        32767
    ).astype(np.int16)

    return AudioSegment(
        samples.tobytes(),
        frame_rate=frame_rate,
        sample_width=2,
        channels=channels
    )


def bit_depth_reduce(audio, bits=8):
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    max_amp = 32767.0
    levels = 2 ** bits

    reduced = np.round(
        samples / max_amp * (levels / 2)
    ) / (levels / 2) * max_amp

    return numpy_to_audiosegment(
        reduced,
        frame_rate,
        channels
    )


def downsample_retro(audio, target_rate=11025):
    original_rate = audio.frame_rate

    crushed = audio.set_frame_rate(
        target_rate
    )

    crushed = crushed.set_frame_rate(
        original_rate
    )

    return crushed


def add_retro_grit(audio, amount=0.015):
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    noise = np.random.uniform(
        -1,
        1,
        samples.shape
    ) * 32767 * amount

    gritty = samples + noise

    return numpy_to_audiosegment(
        gritty,
        frame_rate,
        channels
    )


def soft_clip(audio, drive=1.4):
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    normalized = samples / 32767.0

    clipped = np.tanh(
        normalized * drive
    )

    output = clipped * 32767.0

    return numpy_to_audiosegment(
        output,
        frame_rate,
        channels
    )


def retro_filter(audio, mode):
    if mode == "Soft":
        return audio.low_pass_filter(9000).high_pass_filter(40)

    if mode == "Classic":
        return audio.low_pass_filter(6500).high_pass_filter(60)

    if mode == "Extreme":
        return audio.low_pass_filter(4200).high_pass_filter(90)

    return audio


def convert_to_8bit(audio, mode):
    audio = audio.set_sample_width(2)

    if mode == "Soft":
        bits = 8
        sample_rate = 16000
        noise = 0.004
        drive = 1.15

    elif mode == "Classic":
        bits = 7
        sample_rate = 11025
        noise = 0.010
        drive = 1.35

    else:
        bits = 5
        sample_rate = 8000
        noise = 0.018
        drive = 1.65

    # 전체 곡을 통째로 8-bit 느낌으로 변환
    processed = normalize(audio)

    processed = downsample_retro(
        processed,
        target_rate=sample_rate
    )

    processed = bit_depth_reduce(
        processed,
        bits=bits
    )

    processed = retro_filter(
        processed,
        mode
    )

    processed = add_retro_grit(
        processed,
        amount=noise
    )

    processed = soft_clip(
        processed,
        drive=drive
    )

    processed = processed.fade_in(200)
    processed = processed.fade_out(900)

    return processed


# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

st.markdown('<div class="control-box">', unsafe_allow_html=True)

mode = st.selectbox(
    "8-Bit Conversion Strength",
    [
        "Soft",
        "Classic",
        "Extreme"
    ],
    index=1
)

st.write(
    "Soft = 원곡 보존 / Classic = 게임기 느낌 / Extreme = 강한 8-bit 파괴감"
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    if st.session_state.source_name != uploaded.name:
        st.session_state.converted_bytes = None
        st.session_state.convert_info = None
        st.session_state.source_name = uploaded.name

    st.audio(uploaded)

    if st.button("Convert Whole Song to 8-Bit"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Preparing full track...",
            "Reducing sample rate...",
            "Applying bit depth reduction...",
            "Adding retro texture...",
            "Finalizing 8-bit version..."
        ]

        for i in range(101):
            time.sleep(0.01)
            progress.progress(i)

            if i < 15:
                status.write(steps[0])
            elif i < 30:
                status.write(steps[1])
            elif i < 48:
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

        audio = AudioSegment.from_file(temp_path)

        if len(audio) < 3000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        converted = convert_to_8bit(
            audio,
            mode
        )

        output_path = "pacoel_8bit_full_convert.mp3"

        converted.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.converted_bytes = f.read()

        st.session_state.convert_info = mode

    if st.session_state.converted_bytes:
        st.success("8-Bit Conversion Complete!")

        st.write(f"Mode: {st.session_state.convert_info}")

        st.audio(
            st.session_state.converted_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download 8-Bit Version",
            st.session_state.converted_bytes,
            file_name="pacoel_8bit_full_convert.mp3",
            mime="audio/mpeg",
        )