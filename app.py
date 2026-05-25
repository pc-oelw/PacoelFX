import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup, high_pass_filter
import librosa
import numpy as np
import tempfile
import time
import os
import requests
import replicate

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Pacoel Wave",
    page_icon="🎵",
    layout="wide"
)

# -------------------------
# Replicate token
# -------------------------
try:
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
    REPLICATE_READY = True
except Exception:
    REPLICATE_READY = False

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

.command-box {
    background: #cfcfd4;
    color: #111111;
    padding: 20px;
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

div[data-testid="stAlert"] {
    background-color: #c7c7cc;
    color: #111111;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown(
    '<div class="main-title">🎵 Pacoel Wave</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Command-Based Generative AI Remix Engine</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

Upload a song, write remix commands, and Pacoel Wave will send
part of your track to generative AI so the new intro layer and drop
are based on the original song instead of being random.

</div>
""", unsafe_allow_html=True)

# -------------------------
# Drum samples
# -------------------------
def load_sound(path):
    if os.path.exists(path):
        return AudioSegment.from_file(path)
    return None


kick = load_sound("sounds/kick.wav")
snare = load_sound("sounds/snare.wav")
hihat = load_sound("sounds/hihat.wav")


# -------------------------
# Safe speedup
# -------------------------
def safe_speedup(audio, playback_speed):
    if playback_speed <= 1.01:
        return audio

    try:
        return speedup(
            audio,
            playback_speed=playback_speed,
            chunk_size=90,
            crossfade=20
        )
    except Exception:
        return audio


# -------------------------
# Audio analysis
# -------------------------
def analyze_audio(file_path):
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

    beat_times = librosa.frames_to_time(
        beats,
        sr=sr
    )

    beat_ms = [
        int(t * 1000)
        for t in beat_times
    ]

    rms = librosa.feature.rms(y=y)[0]

    rms_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr
    )

    return int(bpm), beat_ms, rms, rms_times


# -------------------------
# Nearest beat
# -------------------------
def nearest_beat_ms(target_ms, beat_ms):
    if not beat_ms:
        return target_ms

    nearby = [
        b for b in beat_ms
        if abs(b - target_ms) < 2200
    ]

    if not nearby:
        return target_ms

    return min(
        nearby,
        key=lambda x: abs(x - target_ms)
    )


# -------------------------
# Drop point detection
# -------------------------
def find_drop_point_ms(total_ms, rms, rms_times, beat_ms):
    if len(rms) < 10:
        return int(total_ms * 0.62)

    start_sec = (total_ms / 1000) * 0.35
    end_sec = (total_ms / 1000) * 0.78

    candidates = []

    for i in range(5, len(rms)):
        t = rms_times[i]

        if t < start_sec or t > end_sec:
            continue

        before = np.mean(rms[max(0, i - 5):i])
        now = rms[i]
        rise = now - before

        candidates.append((rise, t))

    if not candidates:
        drop_ms = int(total_ms * 0.62)
    else:
        candidates.sort(reverse=True)
        drop_ms = int(candidates[0][1] * 1000)

    drop_ms = nearest_beat_ms(
        drop_ms,
        beat_ms
    )

    drop_ms = max(
        int(total_ms * 0.42),
        min(drop_ms, int(total_ms * 0.82))
    )

    return drop_ms


# -------------------------
# Parse user command
# -------------------------
def parse_command(user_command, detected_bpm):
    command = user_command.lower() if user_command else ""

    target_bpm = 240

    words = command.replace("/", " ").replace(":", " ").split()

    for i, word in enumerate(words):
        if word == "bpm" and i + 1 < len(words):
            try:
                target_bpm = int(words[i + 1])
            except Exception:
                pass

    target_bpm = max(
        180,
        min(target_bpm, 280)
    )

    if "280" in command:
        target_bpm = 280
    elif "270" in command:
        target_bpm = 270
    elif "260" in command:
        target_bpm = 260
    elif "250" in command:
        target_bpm = 250
    elif "240" in command:
        target_bpm = 240
    elif "220" in command:
        target_bpm = 220

    if "sudden" in command or "갑자기" in command:
        drop_type = "sudden explosive tempo switch drop"
    elif "smooth" in command or "자연" in command:
        drop_type = "smooth build into a fast drop"
    else:
        drop_type = "sudden tempo-switch drop"

    if "glitch" in command or "글리치" in command:
        intro_type = "glitchy chopped intro texture"
    elif "dark" in command or "어둡" in command:
        intro_type = "dark electronic intro texture"
    elif "cute" in command or "귀여" in command:
        intro_type = "bright cute arcade synth intro texture"
    else:
        intro_type = "bright arcade electronic intro texture"

    if "hard" in command or "distorted" in command or "강" in command:
        kick_type = "heavy distorted hardcore kicks"
    else:
        kick_type = "fast clean hardcore kicks"

    cleaned_command = command.replace("/", ", ")

    return {
        "target_bpm": target_bpm,
        "drop_type": drop_type,
        "intro_type": intro_type,
        "kick_type": kick_type,
        "raw": cleaned_command
    }


# -------------------------
# Prompt builders
# -------------------------
def make_intro_prompt(command_info):
    prompt = (
        f"{command_info['intro_type']}, "
        "short remix intro layer, energetic rhythm game electronic style, "
        "bright synths, chopped melodic texture, no vocals, instrumental, "
        "designed to layer under the original song, clean mix"
    )

    if command_info["raw"]:
        prompt += ", user direction: " + command_info["raw"]

    return prompt


def make_drop_prompt(command_info):
    target_bpm = command_info["target_bpm"]

    prompt = (
        f"chaotic J-core speedcore drop, {target_bpm} BPM, "
        f"{command_info['drop_type']}, {command_info['kick_type']}, "
        "rapid drum fills, bright arcade synth leads, rhythm game boss song energy, "
        "hyper energetic electronic drop, intense but clean mix, instrumental, no vocals, "
        "melody should follow the input audio"
    )

    if command_info["raw"]:
        prompt += ", user direction: " + command_info["raw"]

    return prompt


# -------------------------
# Export segment to temp file
# -------------------------
def export_segment(segment, suffix=".wav"):
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    segment.export(
        temp_file.name,
        format=suffix.replace(".", "")
    )

    temp_file.close()

    return temp_file.name


# -------------------------
# Replicate MusicGen call
# -------------------------
def generate_ai_audio(prompt, input_audio_path, duration=12, continuation=False):
    try:
        with open(input_audio_path, "rb") as audio_file:
            output = replicate.run(
                "meta/musicgen",
                input={
                    "model_version": "stereo-melody-large",
                    "prompt": prompt,
                    "input_audio": audio_file,
                    "duration": duration,
                    "continuation": continuation,
                    "output_format": "mp3",
                    "temperature": 1.0,
                    "classifier_free_guidance": 4,
                    "top_k": 250,
                    "top_p": 0
                }
            )

        return output

    except Exception as e:
        raise RuntimeError(f"Replicate generation failed: {e}")


# -------------------------
# Download generated audio
# -------------------------
def download_ai_audio(output):
    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    # FileOutput object
    if hasattr(output, "read"):
        temp_audio.write(output.read())
        temp_audio.close()
        return temp_audio.name

    # List output
    if isinstance(output, list):
        first = output[0]

        if hasattr(first, "read"):
            temp_audio.write(first.read())
            temp_audio.close()
            return temp_audio.name

        url = str(first)

    else:
        url = str(output)

    response = requests.get(
        url,
        timeout=180
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download AI audio. Status code: {response.status_code}"
        )

    temp_audio.write(response.content)
    temp_audio.close()

    return temp_audio.name


# -------------------------
# Intro processing
# -------------------------
def process_intro(original_intro, ai_intro_layer, bpm):
    intro = original_intro.fade_in(300)
    intro = high_pass_filter(intro, 35)

    # AI intro layer를 작게 깔아서 초반도 확실히 변하게 함
    if ai_intro_layer:
        ai_layer = ai_intro_layer[:len(intro)]
        ai_layer = ai_layer - 7
        ai_layer = high_pass_filter(ai_layer, 120)

        intro = intro.overlay(
            ai_layer,
            position=0
        )

    beat_ms = int(60000 / max(80, bpm))
    start = int(len(intro) * 0.35)
    end = len(intro) - 900

    for pos in range(start, max(start, end), beat_ms * 2):
        if hihat:
            intro = intro.overlay(
                hihat - 22,
                position=pos + beat_ms // 2
            )

        if kick and pos > len(intro) * 0.70:
            intro = intro.overlay(
                kick - 13,
                position=pos
            )

    return intro


# -------------------------
# Build-up processing
# -------------------------
def process_build(build, bpm):
    if len(build) < 4000:
        return build

    parts = 4
    part_len = len(build) // parts

    speeds = [1.00, 1.03, 1.07, 1.12]
    result = AudioSegment.empty()

    for i in range(parts):
        chunk = build[i * part_len:(i + 1) * part_len]

        chunk = safe_speedup(
            chunk,
            speeds[i]
        )

        chunk = chunk + i

        if len(result) == 0:
            result += chunk
        else:
            result = result.append(
                chunk,
                crossfade=80
            )

    beat_ms = int(60000 / max(90, bpm))
    end_limit = len(result) - 800

    for pos in range(0, max(0, end_limit), beat_ms):
        if hihat:
            result = result.overlay(
                hihat - 18,
                position=pos + beat_ms // 2
            )

        if kick and pos > len(result) * 0.50 and pos % (beat_ms * 2) == 0:
            result = result.overlay(
                kick - 10,
                position=pos
            )

    return result


# -------------------------
# AI drop processing
# -------------------------
def process_ai_drop(ai_drop, target_bpm):
    drop = ai_drop + 2

    beat_ms = int(60000 / target_bpm)
    kick_interval = max(95, beat_ms // 2)

    end_limit = len(drop) - 1500

    for pos in range(0, max(0, end_limit), kick_interval):
        if kick:
            drop = drop.overlay(
                kick + 1,
                position=pos
            )

        if hihat and pos % (kick_interval * 2) == 0:
            drop = drop.overlay(
                hihat - 11,
                position=pos + 25
            )

        if snare and pos % (kick_interval * 4) == 0:
            drop = drop.overlay(
                snare - 7,
                position=pos + 60
            )

        if kick and pos % (kick_interval * 8) == 0:
            extra = pos + int(kick_interval * 0.55)

            if extra < end_limit:
                drop = drop.overlay(
                    kick - 2,
                    position=extra
                )

    drop = drop.fade_in(80)
    drop = drop.fade_out(1200)

    return drop


# -------------------------
# Fallback drop
# -------------------------
def fallback_drop(source_audio, target_bpm):
    drop = safe_speedup(
        source_audio,
        1.15
    )

    return process_ai_drop(
        drop,
        target_bpm
    )


# -------------------------
# Session state
# -------------------------
if "remix_bytes" not in st.session_state:
    st.session_state.remix_bytes = None

if "remix_bpm" not in st.session_state:
    st.session_state.remix_bpm = None

if "drop_bpm" not in st.session_state:
    st.session_state.drop_bpm = None

if "ai_used" not in st.session_state:
    st.session_state.ai_used = None

if "remix_source" not in st.session_state:
    st.session_state.remix_source = None

if "error_text" not in st.session_state:
    st.session_state.error_text = None


# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

st.markdown('<div class="command-box">', unsafe_allow_html=True)

user_command = st.text_area(
    "AI Remix Command",
    value="""/style chaotic j-core speedcore
/bpm 250
/drop sudden tempo switch
/intro glitchy chopped melody
/kick heavy distorted hardcore kick
/vibe rhythm game boss song
""",
    height=150
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    if st.session_state.remix_source != uploaded.name:
        st.session_state.remix_bytes = None
        st.session_state.remix_bpm = None
        st.session_state.drop_bpm = None
        st.session_state.ai_used = None
        st.session_state.error_text = None
        st.session_state.remix_source = uploaded.name

    st.audio(uploaded)

    if not REPLICATE_READY:
        st.warning(
            "REPLICATE_API_TOKEN이 아직 설정되지 않았어. Streamlit Secrets에 먼저 넣어야 해."
        )

    if st.button("Generate Command AI Remix"):
        if not REPLICATE_READY:
            st.error("Replicate API token is not set.")
            st.stop()

        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing BPM and energy...",
            "Finding musical drop point...",
            "Reading remix command...",
            "Preparing original audio reference...",
            "Generating AI intro layer...",
            "Generating AI melody-based drop...",
            "Processing build-up...",
            "Blending remix sections...",
            "Finalizing remix..."
        ]

        for i in range(101):
            time.sleep(0.012)
            progress.progress(i)

            if i < 8:
                status.write(steps[0])
            elif i < 18:
                status.write(steps[1])
            elif i < 28:
                status.write(steps[2])
            elif i < 36:
                status.write(steps[3])
            elif i < 44:
                status.write(steps[4])
            elif i < 58:
                status.write(steps[5])
            elif i < 76:
                status.write(steps[6])
            elif i < 86:
                status.write(steps[7])
            elif i < 96:
                status.write(steps[8])
            else:
                status.write(steps[9])

        file_ext = os.path.splitext(uploaded.name)[1]

        if file_ext.lower() not in [".mp3", ".wav"]:
            file_ext = ".mp3"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            uploaded.seek(0)
            tmp.write(uploaded.read())
            temp_path = tmp.name

        bpm, beat_ms, rms, rms_times = analyze_audio(temp_path)

        original_audio = AudioSegment.from_file(temp_path)

        if len(original_audio) < 15000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        total = len(original_audio)

        drop_start = find_drop_point_ms(
            total,
            rms,
            rms_times,
            beat_ms
        )

        intro_target = int(drop_start * 0.42)

        intro_end = nearest_beat_ms(
            intro_target,
            beat_ms
        )

        intro_end = max(
            4500,
            min(intro_end, int(total * 0.32))
        )

        drop_start = max(
            intro_end + 7000,
            drop_start
        )

        drop_start = min(
            drop_start,
            int(total * 0.82)
        )

        intro = original_audio[:intro_end]
        build = original_audio[intro_end:drop_start]
        original_drop_ref = original_audio[drop_start:]

        command_info = parse_command(
            user_command,
            bpm
        )

        target_bpm = command_info["target_bpm"]

        intro_prompt = make_intro_prompt(
            command_info
        )

        drop_prompt = make_drop_prompt(
            command_info
        )

        intro_ref = original_audio[:min(len(original_audio), 9000)]

        build_ref_start = max(
            0,
            drop_start - 12000
        )

        build_ref = original_audio[
            build_ref_start:drop_start
        ]

        if len(build_ref) < 4000:
            build_ref = original_audio[:min(len(original_audio), 12000)]

        intro_ref_path = export_segment(
            intro_ref,
            suffix=".wav"
        )

        build_ref_path = export_segment(
            build_ref,
            suffix=".wav"
        )

        ai_success = False
        ai_intro_layer = None
        error_text = None

        try:
            status.write("Generating AI intro layer from your song...")

            intro_output = generate_ai_audio(
                intro_prompt,
                intro_ref_path,
                duration=8,
                continuation=False
            )

            intro_ai_path = download_ai_audio(
                intro_output
            )

            ai_intro_layer = AudioSegment.from_file(
                intro_ai_path
            )

            status.write("Generating melody-based AI speedcore drop...")

            drop_output = generate_ai_audio(
                drop_prompt,
                build_ref_path,
                duration=18,
                continuation=False
            )

            drop_ai_path = download_ai_audio(
                drop_output
            )

            ai_drop = AudioSegment.from_file(
                drop_ai_path
            )

            ai_drop = process_ai_drop(
                ai_drop,
                target_bpm
            )

            ai_success = True

        except Exception as e:
            error_text = str(e)

            st.warning("AI generation failed. Using fallback remix mode.")
            st.caption(error_text)

            ai_intro_layer = None

            ai_drop = fallback_drop(
                original_drop_ref,
                target_bpm
            )

            ai_success = False

        intro = process_intro(
            intro,
            ai_intro_layer,
            bpm
        )

        build = process_build(
            build,
            bpm
        )

        pre_drop = AudioSegment.silent(
            duration=240
        )

        remix = intro.append(
            build,
            crossfade=220
        )

        remix = remix.append(
            pre_drop,
            crossfade=5
        )

        remix = remix.append(
            ai_drop,
            crossfade=90
        )

        remix = remix.fade_out(1200)

        max_len = len(original_audio) + 4000

        if len(remix) > max_len:
            remix = remix[:max_len]

        output_path = "pacoel_command_ai_remix.mp3"

        remix.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.remix_bytes = f.read()

        st.session_state.remix_bpm = bpm
        st.session_state.drop_bpm = target_bpm
        st.session_state.ai_used = ai_success
        st.session_state.error_text = error_text

    if st.session_state.remix_bytes:
        st.write(f"Detected BPM: {st.session_state.remix_bpm}")
        st.write(f"AI Target Drop BPM: {st.session_state.drop_bpm}")

        if st.session_state.ai_used:
            st.success("Generative AI used your audio as a reference.")
        else:
            st.info("Fallback remix mode was used.")

            if st.session_state.error_text:
                st.caption(st.session_state.error_text)

        st.success("Remix Complete!")

        st.audio(
            st.session_state.remix_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download Remix",
            st.session_state.remix_bytes,
            file_name="pacoel_command_ai_remix.mp3",
            mime="audio/mpeg",
        )