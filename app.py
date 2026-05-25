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
    '<div class="sub-title">True Retro 8-Bit Converter</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

Upload a song and convert the whole track into a stronger retro 8-bit style.
This version does not add new melodies or drums. It reshapes the whole audio
with waveform quantization, sample-rate reduction, square-like distortion,
and console-style filtering.

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


def bit_depth_reduce(audio, bits=6):
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


def sample_hold(audio, hold=4):
    """
    샘플을 일정 개수만큼 붙잡아서 계단식 디지털 느낌을 만듦.
    이게 기존 버전보다 8-bit 느낌을 더 강하게 냄.
    """
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    if channels == 2:
        result = samples.copy()

        for ch in range(2):
            channel = result[:, ch]
            held = channel.copy()

            for i in range(0, len(channel), hold):
                held[i:i + hold] = channel[i]

            result[:, ch] = held

    else:
        result = samples.copy()

        for i in range(0, len(samples), hold):
            result[i:i + hold] = samples[i]

    return numpy_to_audiosegment(
        result,
        frame_rate,
        channels
    )


def square_shape(audio, amount=0.55):
    """
    원곡 파형을 약간 사각파처럼 바꿈.
    amount가 높을수록 게임기 느낌이 강해짐.
    """
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    normalized = samples / 32767.0

    squared = np.sign(normalized) * np.sqrt(
        np.abs(normalized)
    )

    mixed = normalized * (1 - amount) + squared * amount

    output = mixed * 32767.0

    return numpy_to_audiosegment(
        output,
        frame_rate,
        channels
    )


def stair_step_distortion(audio, steps=18, mix=0.65):
    """
    파형을 계단식으로 만들어서 진짜 8-bit스럽게 만듦.
    """
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    normalized = samples / 32767.0

    stepped = np.round(
        normalized * steps
    ) / steps

    mixed = normalized * (1 - mix) + stepped * mix

    output = mixed * 32767.0

    return numpy_to_audiosegment(
        output,
        frame_rate,
        channels
    )


def downsample_hard(audio, target_rate=9000):
    original_rate = audio.frame_rate

    lowered = audio.set_frame_rate(
        target_rate
    )

    restored = lowered.set_frame_rate(
        original_rate
    )

    return restored


def add_tiny_digital_noise(audio, amount=0.006):
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    noise = np.random.uniform(
        -1,
        1,
        samples.shape
    ) * 32767 * amount

    output = samples + noise

    return numpy_to_audiosegment(
        output,
        frame_rate,
        channels
    )


def hard_clip(audio, drive=1.5):
    samples, frame_rate, channels = audiosegment_to_numpy(audio)

    normalized = samples / 32767.0

    driven = normalized * drive

    clipped = np.clip(
        driven,
        -0.78,
        0.78
    )

    output = clipped / 0.78 * 32767.0

    return numpy_to_audiosegment(
        output,
        frame_rate,
        channels
    )


def retro_console_eq(audio, mode):
    """
    먹먹하지 않게 너무 과한 로우패스는 피하고,
    저음 뭉침을 줄여서 게임기 스피커 같은 느낌을 만듦.
    """
    if mode == "Light":
        result = audio.high_pass_filter(80)
        result = result.low_pass_filter(14000)
        return result

    if mode == "Gameboy":
        result = audio.high_pass_filter(120)
        result = result.low_pass_filter(10000)
        return result

    if mode == "NES":
        result = audio.high_pass_filter(160)
        result = result.low_pass_filter(8500)
        return result

    if mode == "Broken Console":
        result = audio.high_pass_filter(220)
        result = result.low_pass_filter(6500)
        return result

    return audio


def add_brightness(original, processed, mode):
    """
    원곡의 고역 일부를 아주 작게 섞어서 답답함 방지.
    """
    if mode == "Light":
        gain = -12
    elif mode == "Gameboy":
        gain = -14
    elif mode == "NES":
        gain = -16
    else:
        gain = -18

    bright = original.high_pass_filter(4000)
    bright = bright + gain

    return processed.overlay(
        bright
    )


def mono_console(audio, strength=0.8):
    """
    8비트 게임기 느낌을 위해 스테레오를 약하게 모노화.
    """
    if audio.channels == 1:
        return audio

    mono = audio.set_channels(1).set_channels(2)

    original = audio - int(strength * 5)
    mono = mono - int((1 - strength) * 5)

    return original.overlay(
        mono
    )


def convert_to_true_8bit(audio, mode):
    audio = audio.set_sample_width(2)

    original = normalize(audio)

    if mode == "Light":
        bits = 7
        sample_rate = 16000
        hold = 2
        square_amount = 0.30
        stair_mix = 0.40
        steps = 32
        noise = 0.002
        drive = 1.15
        mono_strength = 0.35

    elif mode == "Gameboy":
        bits = 6
        sample_rate = 11025
        hold = 4
        square_amount = 0.55
        stair_mix = 0.65
        steps = 18
        noise = 0.005
        drive = 1.40
        mono_strength = 0.65

    elif mode == "NES":
        bits = 5
        sample_rate = 9000
        hold = 5
        square_amount = 0.70
        stair_mix = 0.78
        steps = 14
        noise = 0.007
        drive = 1.55
        mono_strength = 0.80

    else:
        bits = 4
        sample_rate = 7000
        hold = 7
        square_amount = 0.82
        stair_mix = 0.88
        steps = 10
        noise = 0.011
        drive = 1.85
        mono_strength = 0.95

    processed = original

    processed = mono_console(
        processed,
        strength=mono_strength
    )

    processed = downsample_hard(
        processed,
        target_rate=sample_rate
    )

    processed = sample_hold(
        processed,
        hold=hold
    )

    processed = bit_depth_reduce(
        processed,
        bits=bits
    )

    processed = stair_step_distortion(
        processed,
        steps=steps,
        mix=stair_mix
    )

    processed = square_shape(
        processed,
        amount=square_amount
    )

    processed = hard_clip(
        processed,
        drive=drive
    )

    processed = retro_console_eq(
        processed,
        mode
    )

    processed = add_tiny_digital_noise(
        processed,
        amount=noise
    )

    processed = add_brightness(
        original,
        processed,
        mode
    )

    processed = normalize(processed)

    processed = processed.fade_in(80)
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
    "8-Bit Mode",
    [
        "Light",
        "Gameboy",
        "NES",
        "Broken Console"
    ],
    index=1
)

st.write(
    "Light = 약한 레트로 / Gameboy = 추천 / NES = 더 강한 8비트 / Broken Console = 많이 깨짐"
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    if st.session_state.source_name != uploaded.name:
        st.session_state.converted_bytes = None
        st.session_state.convert_info = None
        st.session_state.source_name = uploaded.name

    st.audio(uploaded)

    if st.button("Convert Whole Song to True 8-Bit"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Preparing full track...",
            "Applying console mono...",
            "Reducing sample rate...",
            "Creating stair-step waveform...",
            "Adding square-wave texture...",
            "Finalizing 8-bit version..."
        ]

        for i in range(101):
            time.sleep(0.008)
            progress.progress(i)

            if i < 12:
                status.write(steps[0])
            elif i < 25:
                status.write(steps[1])
            elif i < 38:
                status.write(steps[2])
            elif i < 52:
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

        audio = AudioSegment.from_file(temp_path)

        if len(audio) < 3000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        converted = convert_to_true_8bit(
            audio,
            mode
        )

        output_path = "pacoel_true_8bit_convert.mp3"

        converted.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.converted_bytes = f.read()

        st.session_state.convert_info = mode

    if st.session_state.converted_bytes:
        st.success("True 8-Bit Conversion Complete!")

        st.write(f"Mode: {st.session_state.convert_info}")

        st.audio(
            st.session_state.converted_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download True 8-Bit Version",
            st.session_state.converted_bytes,
            file_name="pacoel_true_8bit_convert.mp3",
            mime="audio/mpeg",
        )