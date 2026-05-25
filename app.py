import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup
import librosa
import numpy as np
import tempfile
import time
import os

# -------------------------
# 페이지 설정
# -------------------------
st.set_page_config(
    page_title="Pacoel Wave",
    page_icon="🎵",
    layout="wide"
)

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>

.stApp {
    background-color: #e7e7ea;
}

.main-title {
    font-size: 60px;
    font-weight: 900;
    color: #111111;
}

.sub-title {
    font-size: 18px;
    color: #666666;
    margin-bottom: 35px;
}

.info-box {
    background: white;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

.stButton button {
    background-color: black;
    color: white;
    border-radius: 16px;
    height: 54px;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stDownloadButton button {
    background-color: black;
    color: white;
    border-radius: 16px;
    height: 54px;
    width: 100%;
    font-size: 18px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# 제목
# -------------------------
st.markdown(
    '<div class="main-title">🎵 Pacoel Wave</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Adaptive Speedcore Remix Engine</div>',
    unsafe_allow_html=True
)

# -------------------------
# 설명
# -------------------------
st.markdown("""
<div class="info-box">

The AI analyzes your uploaded song and creates
a structured speedcore-inspired remix while
preserving the original music feeling.

</div>
""", unsafe_allow_html=True)

# -------------------------
# 드럼 로드
# -------------------------
def load_sound(path):

    if os.path.exists(path):
        return AudioSegment.from_file(path)

    return None

kick = load_sound("sounds/kick.wav")
snare = load_sound("sounds/snare.wav")
hihat = load_sound("sounds/hihat.wav")

# -------------------------
# BPM 기반 드럼
# -------------------------
def add_drums(audio, bpm, section="build"):

    output = audio

    beat_ms = int(60000 / bpm)

    if section == "intro":

        return output

    elif section == "build":

        interval = beat_ms

        for pos in range(0, len(audio), interval):

            if kick:
                output = output.overlay(
                    kick - 7,
                    position=pos
                )

            if hihat:
                output = output.overlay(
                    hihat - 15,
                    position=pos + 70
                )

    elif section == "drop":

        interval = max(120, beat_ms // 2)

        for pos in range(0, len(audio), interval):

            if kick:
                output = output.overlay(
                    kick + 1,
                    position=pos
                )

            if hihat:
                output = output.overlay(
                    hihat - 9,
                    position=pos + 40
                )

            if snare and pos % (beat_ms * 2) < 50:
                output = output.overlay(
                    snare - 4,
                    position=pos + 90
                )

    return output

# -------------------------
# 업로드
# -------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

# -------------------------
# 실행
# -------------------------
if uploaded:

    st.audio(uploaded)

    if st.button("Generate Remix"):

        progress = st.progress(0)

        status = st.empty()

        steps = [
            "Analyzing Audio...",
            "Detecting BPM...",
            "Finding Song Structure...",
            "Creating Build-up...",
            "Generating Drop...",
            "Adding Drums...",
            "Finalizing..."
        ]

        for i in range(101):

            time.sleep(0.02)

            progress.progress(i)

            if i < 15:
                status.write(steps[0])

            elif i < 30:
                status.write(steps[1])

            elif i < 45:
                status.write(steps[2])

            elif i < 65:
                status.write(steps[3])

            elif i < 85:
                status.write(steps[4])

            elif i < 95:
                status.write(steps[5])

            else:
                status.write(steps[6])

        # -------------------------
        # 임시 저장
        # -------------------------
        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(uploaded.read())

            temp_path = tmp.name

        # -------------------------
        # BPM 분석
        # -------------------------
        y, sr = librosa.load(
            temp_path,
            sr=None
        )

        tempo, beats = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        # -------------------------
        # 오디오 로드
        # -------------------------
        audio = AudioSegment.from_file(temp_path)

        total = len(audio)

        # -------------------------
        # 구조 분리
        # -------------------------
        intro_len = int(total * 0.35)

        build_len = int(total * 0.30)

        intro = audio[:intro_len]

        build = audio[
            intro_len:intro_len + build_len
        ]

        drop = audio[
            intro_len + build_len:
        ]

        # -------------------------
        # 인트로
        # 거의 유지
        # -------------------------
        intro = intro.fade_in(700)

        # -------------------------
        # 빌드업
        # -------------------------
        if bpm < 100:

            build_speed = 1.08
            drop_speed = 1.30

        elif bpm < 140:

            build_speed = 1.05
            drop_speed = 1.22

        else:

            build_speed = 1.03
            drop_speed = 1.15

        build = speedup(
            build,
            playback_speed=build_speed
        )

        build = build + 2

        # -------------------------
        # 드랍
        # -------------------------
        drop = speedup(
            drop,
            playback_speed=drop_speed
        )

        drop = drop + 5

        # -------------------------
        # 드럼 추가
        # -------------------------
        build = add_drums(
            build,
            bpm * build_speed,
            section="build"
        )

        drop = add_drums(
            drop,
            bpm * drop_speed * 1.6,
            section="drop"
        )

        # -------------------------
        # 드랍 직전 정적
        # -------------------------
        silence = AudioSegment.silent(
            duration=180
        )

        # -------------------------
        # 연결
        # -------------------------
        remix = intro.append(
            build,
            crossfade=350
        )

        remix = remix.append(
            silence,
            crossfade=10
        )

        remix = remix.append(
            drop,
            crossfade=120
        )

        # -------------------------
        # 마스터
        # -------------------------
        remix = remix + 1

        # -------------------------
        # 저장
        # -------------------------
        output_path = "pacoel_remix.mp3"

        remix.export(
            output_path,
            format="mp3"
        )

        st.success("Remix Complete!")

        st.audio(output_path)

        with open(output_path, "rb") as f:

            st.download_button(
                "⬇ Download Remix",
                f,
                file_name="pacoel_remix.mp3"
            )
