import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup
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
    '<div class="sub-title">Speedcore Remix Generator</div>',
    unsafe_allow_html=True
)

# -------------------------
# 드럼 파일 로드
# -------------------------
def load_sound(path):

    if os.path.exists(path):
        return AudioSegment.from_file(path)

    return None

kick = load_sound("sounds/kick.wav")
snare = load_sound("sounds/snare.wav")
hihat = load_sound("sounds/hihat.wav")

# -------------------------
# 드럼 추가
# -------------------------
def add_drums(audio, bpm=170):

    beat = int(60000 / bpm)

    output = audio

    for i in range(0, len(audio), beat):

        # 킥
        if kick:
            output = output.overlay(
                kick - 2,
                position=i
            )

        # 하이햇
        if hihat:
            output = output.overlay(
                hihat - 10,
                position=i + beat // 2
            )

        # 스네어
        if snare:
            output = output.overlay(
                snare - 5,
                position=i + beat
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

        for i in range(101):

            time.sleep(0.02)

            progress.progress(i)

            status.write(f"Generating Speedcore... {i}%")

        # -------------------------
        # 임시 저장
        # -------------------------
        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(uploaded.read())

            temp_path = tmp.name

        audio = AudioSegment.from_file(temp_path)

        # -------------------------
        # 구간 분리
        # -------------------------
        length = len(audio)

        first = audio[:length // 2]

        second = audio[length // 2:]

        # -------------------------
        # 초반
        # -------------------------
        first = speedup(
            first,
            playback_speed=1.05
        )

        first = add_drums(
            first,
            bpm=165
        )

        # -------------------------
        # 후반 더 빠르게
        # -------------------------
        second = speedup(
            second,
            playback_speed=1.12
        )

        second = add_drums(
            second,
            bpm=185
        )

        second = second + 3

        # -------------------------
        # 합치기
        # -------------------------
        remixed = first.append(
            second,
            crossfade=250
        )

        # -------------------------
        # 저장
        # -------------------------
        output_path = "speedcore_remix.mp3"

        remixed.export(
            output_path,
            format="mp3"
        )

        st.success("Speedcore Remix Complete!")

        st.audio(output_path)

        with open(output_path, "rb") as f:

            st.download_button(
                "⬇ Download Remix",
                f,
                file_name="speedcore_remix.mp3"
            )
