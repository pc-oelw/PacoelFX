import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup
import librosa
import soundfile as sf
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
    font-size: 58px;
    font-weight: 800;
    color: #111111;
}

.sub-title {
    font-size: 18px;
    color: #666666;
    margin-bottom: 35px;
}

.stButton button {
    background-color: black;
    color: white;
    border-radius: 16px;
    height: 52px;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stDownloadButton button {
    background-color: black;
    color: white;
    border-radius: 16px;
    height: 52px;
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
    '<div class="sub-title">AI Speedcore Remix Generator</div>',
    unsafe_allow_html=True
)

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
# AI 드럼 추가
# -------------------------
def add_ai_drums(audio, beat_times):

    output = audio

    for i, beat in enumerate(beat_times):

        pos = int(beat * 1000)

        # 킥
        if kick and i % 2 == 0:
            output = output.overlay(
                kick - 2,
                position=pos
            )

        # 하이햇
        if hihat:
            output = output.overlay(
                hihat - 12,
                position=pos + 120
            )

        # 스네어
        if snare and i % 4 == 2:
            output = output.overlay(
                snare - 6,
                position=pos
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

    if st.button("Generate AI Remix"):

        progress = st.progress(0)

        status = st.empty()

        # -------------------------
        # 진행 UI
        # -------------------------
        loading_texts = [
            "Analyzing BPM...",
            "Detecting Beats...",
            "Generating Speedcore...",
            "Adding Drums...",
            "Finalizing Remix..."
        ]

        for i in range(101):

            time.sleep(0.025)

            progress.progress(i)

            if i < 20:
                status.write(loading_texts[0])

            elif i < 40:
                status.write(loading_texts[1])

            elif i < 70:
                status.write(loading_texts[2])

            elif i < 90:
                status.write(loading_texts[3])

            else:
                status.write(loading_texts[4])

        # -------------------------
        # 임시 저장
        # -------------------------
        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(uploaded.read())

            temp_path = tmp.name

        # -------------------------
        # librosa 분석
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
        # BPM 기반 속도
        # -------------------------
        if tempo < 100:
            speed = 1.15

        elif tempo < 130:
            speed = 1.10

        else:
            speed = 1.05

        # -------------------------
        # pydub 로드
        # -------------------------
        audio = AudioSegment.from_file(temp_path)

        # -------------------------
        # 구간별 처리
        # -------------------------
        length = len(audio)

        first = audio[:length // 2]

        second = audio[length // 2:]

        # 초반
        first = speedup(
            first,
            playback_speed=speed
        )

        # 후반 더 빠르게
        second = speedup(
            second,
            playback_speed=speed + 0.05
        )

        # 합치기
        remixed = first.append(
            second,
            crossfade=250
        )

        # -------------------------
        # AI 드럼 추가
        # -------------------------
        remixed = add_ai_drums(
            remixed,
            beat_times
        )

        # -------------------------
        # 볼륨
        # -------------------------
        remixed = remixed + 2

        # -------------------------
        # 저장
        # -------------------------
        output_path = "ai_speedcore_remix.mp3"

        remixed.export(
            output_path,
            format="mp3"
        )

        st.success("AI Remix Complete!")

        st.audio(output_path)

        with open(output_path, "rb") as f:

            st.download_button(
                "⬇ Download Remix",
                f,
                file_name="ai_speedcore_remix.mp3"
            )
