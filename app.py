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
    '<div class="sub-title">Sharp 8-Bit Audio Converter</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

Upload a song and convert the whole track into a sharper retro 8-bit style.
This version avoids the muffled sound by keeping more high frequencies
and adding digital crunch instead of simply cutting the audio.

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
# Audio helpers
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


def sample_rate_reduce(audio, target_rate=12000):
    original_rate = audio.frame_rate

    reduced = audio.set_frame_rate(
        target_rate
    )

    restored = reduced.set_frame_rate(
        original_rate
    )

    return restored


def add_digital_noise(audio, amount=0.006):
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    noise = np.random.uniform(
        -1,
        1,
        samples.shape
    ) * 32767 * amount

    result = samples + noise

    return numpy_to_audiosegment(
        result,
        frame_rate,
        channels
    )


def hard_clip(audio, drive=1.35):
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    normalized = samples / 32767.0

    driven = normalized * drive

    clipped = np.clip(
        driven,
        -0.85,
        0.85
    )

    output = clipped / 0.85 * 32767.0

    return numpy_to_audiosegment(
        output,
        frame_rate,
        channels
    )


def add_brightness(original, processed, strength=0.45):
    # 원본에서 고음 느낌만 약하게 가져와서 먹먹함 방지
    bright = original.high_pass_filter(3500)

    bright = bright - int(12 - strength * 10)

    result = processed.overlay(
        bright
    )

    return result


def retro_eq(audio, mode):
    # 예전 버전처럼 고음을 세게 자르지 않음
    if mode == "Soft":
        result = audio.high_pass_filter(35)
        result = result.low_pass_filter(15500)
        return result

    if mode == "Classic":
        result = audio.high_pass_filter(45)
        result = result.low_pass_filter(13500)
        return result

    if mode == "Extreme":
        result = audio.high_pass_filter(65)
        result = result.low_pass_filter(11500)
        return result

    return audio


def convert_to_8bit(audio, mode):
    audio = audio.set_sample_width(2)

    original = normalize(audio)

    if mode == "Soft":
        bits = 8
        sample_rate = 16000
        noise = 0.003
        drive = 1.15
        brightness = 0.35

    elif mode == "Classic":
        bits = 7
        sample_rate = 12000
        noise = 0.006
        drive = 1.35
        brightness = 0.50

    else:
        bits = 6
        sample_rate = 9000
        noise = 0.010
        drive = 1.55
        brightness = 0.65

    processed = original

    # 1. 샘플레이트 낮춰서 레트로 디지털 질감
    processed = sample_rate_reduce(
        processed,
        target_rate=sample_rate
    )

    # 2. 비트 깊이 감소
    processed = bit_depth_reduce(
        processed,
        bits=bits
    )

    # 3. 먹먹하지 않게 EQ는 약하게
    processed = retro_eq(
        processed,
        mode
    )

    # 4. 디지털 노이즈
    processed = add_digital_noise(
        processed,
        amount=noise
    )

    # 5. 약한 하드 클리핑으로 게임기 같은 날카로움
    processed = hard_clip(
        processed,
        drive=drive
    )

    # 6. 원곡 고음 일부를 섞어서 먹먹함 방지
    processed = add_brightness(
        original,
        processed,
        strength=brightness
    )

    processed = normalize(processed)

    processed = processed.fade_in(120)
    processed = processed.fade_out(700)

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
    "Soft = 원곡 보존 / Classic = 선명한 8-bit / Extreme = 더 거친 디지털 깨짐"
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    if st.session_state.source_name != uploaded.name:
        st.session_state.converted_bytes = None
        st.session_state.convert_info = None
        st.session_state.source_name = uploaded.name

    st.audio(uploaded)

    if st.button("Convert Whole Song to Sharp 8-Bit"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Preparing full track...",
            "Reducing sample rate...",
            "Reducing bit depth...",
            "Adding digital crunch...",
            "Restoring brightness...",
            "Finalizing 8-bit version..."
        ]

        for i in range(101):
            time.sleep(0.008)
            progress.progress(i)

            if i < 12:
                status.write(steps[0])
            elif i < 25:
                status.write(steps[1])
            elif i < 40:
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

        audio = AudioSegment.from_file(temp_path)

        if len(audio) < 3000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        converted = convert_to_8bit(
            audio,
            mode
        )

        output_path = "pacoel_sharp_8bit_convert.mp3"

        converted.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.converted_bytes = f.read()

        st.session_state.convert_info = mode

    if st.session_state.converted_bytes:
        st.success("Sharp 8-Bit Conversion Complete!")

        st.write(f"Mode: {st.session_state.convert_info}")

        st.audio(
            st.session_state.converted_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download Sharp 8-Bit Version",
            st.session_state.converted_bytes,
            file_name="pacoel_sharp_8bit_convert.mp3",
            mime="audio/mpeg",
        )