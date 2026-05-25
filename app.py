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
    '<div class="sub-title">AI Speedcore Remix Engine</div>',
    unsafe_allow_html=True
)

# -------------------------
# 설명
# -------------------------
st.markdown("""
<div class="info-box">

AI analyzes the uploaded track and automatically creates
a progressive speedcore remix with intro preservation,
build-up transitions, drop enhancement, and adaptive drums.

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
# 드럼 함수
# -------------------------
def add_drums(audio, beat_times, section="intro"):

    output = audio

    for i, beat in enumerate(beat_times):

        pos = int(beat * 1000)

        # -------------------------
        # 인트로
        # -------------------------
        if section == "intro":

            if hihat and i % 8 == 0:

                output = output.overlay(
                    hihat - 18,
                    position=pos
                )

        # -------------------------
        # 빌드업
        # -------------------------
        elif section == "build":

            if kick and i % 4 == 0:

                output = output.overlay(
                    kick - 5,
                    position=pos
                )

            if hihat:

                output = output.overlay(
                    hihat - 14,
                    position=pos + 60
                )

        # -------------------------
        # 드랍
        # -------------------------
        elif section == "drop":

            if kick:

                output = output.overlay(
                    kick + 1,
                    position=pos
                )

            if snare and i % 2 == 0:

                output = output.overlay(
                    snare - 4,
                    position=pos + 40
                )

            if hihat:

                output = output.overlay(
                    hihat - 10,
                    position=pos + 20
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

    if st.button("Generate AI Speedcore Remix"):

        progress = st.progress(0)

        status = st.empty()

        steps = [
            "Analyzing BPM...",
            "Detecting Beats...",
            "Building Intro...",
            "Generating Build-up...",
            "Generating Drop...",
            "Adding Drums...",
            "Finalizing Remix..."
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

        beat_times = librosa.frames_to_time(
            beats,
            sr=sr
        )

        # -------------------------
        # 오디오 로드
        # -------------------------
        audio = AudioSegment.from_file(temp_path)

        total = len(audio)

        # -------------------------
        # 구간 분리
        # -------------------------
        intro_len = int(total * 0.20)
        build_len = int(total * 0.35)

        intro = audio[:intro_len]

        build = audio[
            intro_len:intro_len + build_len
        ]

        drop = audio[
            intro_len + build_len:
        ]

        # -------------------------
        # 스피드코어 BPM 느낌
        # -------------------------
        if tempo < 100:

            build_speed = 1.25
            drop_speed = 1.45

        elif tempo < 140:

            build_speed = 1.18
            drop_speed = 1.35

        else:

            build_speed = 1.10
            drop_speed = 1.25

        # -------------------------
        # 인트로
        # -------------------------
        intro = intro.fade_in(500)

        # -------------------------
        # 빌드업
        # -------------------------
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

        drop = drop + 7

        # -------------------------
        # 드럼 추가
        # -------------------------
        intro = add_drums(
            intro,
            beat_times,
            section="intro"
        )

        build = add_drums(
            build,
            beat_times,
            section="build"
        )

        drop = add_drums(
            drop,
            beat_times,
            section="drop"
        )

        # -------------------------
        # 연결
        # -------------------------
        remix = intro.append(
            build,
            crossfade=400
        )

        remix = remix.append(
            drop,
            crossfade=500
        )

        # -------------------------
        # 마스터 볼륨
        # -------------------------
        remix = remix + 1

        # -------------------------
        # 저장
        # -------------------------
        output_path = "pacoel_speedcore_remix.mp3"

        remix.export(
            output_path,
            format="mp3"
        )

        st.success("AI Speedcore Remix Complete!")

        st.audio(output_path)

        with open(output_path, "rb") as f:

            st.download_button(
                "⬇ Download Remix",
                f,
                file_name="pacoel_speedcore_remix.mp3"
            )
