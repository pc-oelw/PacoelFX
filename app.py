import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup
import librosa
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
    '<div class="sub-title">AI Adaptive Remix</div>',
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
                hihat - 14,
                position=pos + 100
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

        loading_steps = [
            "Analyzing Audio...",
            "Detecting BPM...",
            "Detecting Beats...",
            "Generating Remix...",
            "Adding Drums...",
            "Finalizing..."
        ]

        for i in range(101):

            time.sleep(0.025)

            progress.progress(i)

            if i < 15:
                status.write(loading_steps[0])

            elif i < 30:
                status.write(loading_steps[1])

            elif i < 50:
                status.write(loading_steps[2])

            elif i < 75:
                status.write(loading_steps[3])

            elif i < 90:
                status.write(loading_steps[4])

            else:
                status.write(loading_steps[5])

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
        # BPM 기반 속도 계산
        # -------------------------
        if tempo < 90:

            first_speed = 1.12
            second_speed = 1.20

        elif tempo < 120:

            first_speed = 1.08
            second_speed = 1.14

        else:

            first_speed = 1.03
            second_speed = 1.08

        # -------------------------
        # 오디오 로드
        # -------------------------
        audio = AudioSegment.from_file(temp_path)

        length = len(audio)

        # -------------------------
        # 구간 분리
        # -------------------------
        first = audio[:length // 2]

        second = audio[length // 2:]

        # -------------------------
        # 초반
        # -------------------------
        first = speedup(
            first,
            playback_speed=first_speed
        )

        # -------------------------
        # 후반
        # -------------------------
        second = speedup(
            second,
            playback_speed=second_speed
        )

        # -------------------------
        # 합치기
        # -------------------------
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
        # 후반 강화
        # -------------------------
        remixed = remixed + 2

        # -------------------------
        # 저장
        # -------------------------
        output_path = "ai_remix.mp3"

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
                file_name="ai_remix.mp3"
            )
