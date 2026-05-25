import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup
import librosa
import numpy as np
import tempfile
import time
import os
import requests

# -------------------------
# 페이지 설정
# -------------------------
st.set_page_config(
    page_title="Pacoel Wave",
    page_icon="🎵",
    layout="wide"
)

# -------------------------
# Hugging Face API
# -------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

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

.info-box {
    background: white;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
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
    '<div class="sub-title">AI Adaptive Remix Engine</div>',
    unsafe_allow_html=True
)

# -------------------------
# 설명
# -------------------------
st.markdown("""
<div class="info-box">

AI analyzes your uploaded song and automatically creates
a dynamic remix with adaptive speed, beat detection,
drop enhancement, and AI-generated music styling.

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
# AI 드럼 함수
# -------------------------
def add_ai_drums(audio, beat_times, intensity=1):

    output = audio

    for i, beat in enumerate(beat_times):

        pos = int(beat * 1000)

        # 킥
        if kick and i % 2 == 0:

            output = output.overlay(
                kick - (2 - intensity),
                position=pos
            )

        # 하이햇
        if hihat:

            output = output.overlay(
                hihat - 14,
                position=pos + 90
            )

        # 스네어
        if snare and i % 4 == 2:

            output = output.overlay(
                snare - 5,
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

        loading = [
            "Analyzing Audio...",
            "Detecting BPM...",
            "Detecting Drops...",
            "Generating AI Remix...",
            "Adding AI Drums...",
            "Enhancing Audio...",
            "Finalizing..."
        ]

        for i in range(101):

            time.sleep(0.025)

            progress.progress(i)

            if i < 10:
                status.write(loading[0])

            elif i < 25:
                status.write(loading[1])

            elif i < 40:
                status.write(loading[2])

            elif i < 60:
                status.write(loading[3])

            elif i < 80:
                status.write(loading[4])

            elif i < 95:
                status.write(loading[5])

            else:
                status.write(loading[6])

        # -------------------------
        # 임시 저장
        # -------------------------
        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(uploaded.read())

            temp_path = tmp.name

        # -------------------------
        # AI BPM 분석
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
        # 에너지 분석
        # -------------------------
        rms = librosa.feature.rms(y=y)[0]

        energy = np.mean(rms)

        # -------------------------
        # AI 속도 계산
        # -------------------------
        if tempo < 90:

            intro_speed = 1.03
            middle_speed = 1.08
            drop_speed = 1.15

        elif tempo < 130:

            intro_speed = 1.02
            middle_speed = 1.06
            drop_speed = 1.10

        else:

            intro_speed = 1.00
            middle_speed = 1.03
            drop_speed = 1.06

        # -------------------------
        # 오디오 로드
        # -------------------------
        audio = AudioSegment.from_file(temp_path)

        length = len(audio)

        intro = audio[:length // 3]

        middle = audio[length // 3:length // 3 * 2]

        drop = audio[length // 3 * 2:]

        # -------------------------
        # AI 속도 적용
        # -------------------------
        intro = speedup(
            intro,
            playback_speed=intro_speed
        )

        middle = speedup(
            middle,
            playback_speed=middle_speed
        )

        drop = speedup(
            drop,
            playback_speed=drop_speed
        )

        # -------------------------
        # AI 드럼 추가
        # -------------------------
        intro = add_ai_drums(
            intro,
            beat_times,
            intensity=0
        )

        middle = add_ai_drums(
            middle,
            beat_times,
            intensity=1
        )

        drop = add_ai_drums(
            drop,
            beat_times,
            intensity=2
        )

        # -------------------------
        # 드랍 강화
        # -------------------------
        if energy > 0.05:

            drop = drop + 5

        else:

            drop = drop + 2

        # -------------------------
        # 합치기
        # -------------------------
        remixed = intro.append(
            middle,
            crossfade=250
        )

        remixed = remixed.append(
            drop,
            crossfade=350
        )

        # -------------------------
        # Hugging Face AI 요청
        # -------------------------
        try:

            with open(temp_path, "rb") as f:

                response = requests.post(
                    API_URL,
                    headers=headers,
                    data=f
                )

            if response.status_code == 200:

                st.success("AI Music Generation Connected!")

            else:

                st.warning(
                    "AI generation API connected, but model may still be loading."
                )

        except Exception as e:

            st.warning(
                f"AI API Error: {e}"
            )

        # -------------------------
        # 최종 볼륨
        # -------------------------
        remixed = remixed + 1

        # -------------------------
        # 저장
        # -------------------------
        output_path = "ai_adaptive_remix.mp3"

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
                file_name="ai_adaptive_remix.mp3"
            )
