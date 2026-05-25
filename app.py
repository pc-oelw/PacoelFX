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
    margin-bottom: 0px;
}

.sub-title {
    font-size: 20px;
    color: #666666;
    margin-top: -10px;
    margin-bottom: 35px;
}

.upload-box {
    background: white;
    padding: 30px;
    border-radius: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
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
    '<div class="sub-title">Automatic Remix Generator</div>',
    unsafe_allow_html=True
)

# -------------------------
# 드럼 파일 불러오기
# -------------------------
def load_sound(path):

    if os.path.exists(path):
        return AudioSegment.from_file(path)

    return None

kick = load_sound("sounds/kick.wav")
snare = load_sound("sounds/snare.wav")
hihat = load_sound("sounds/hihat.wav")

# -------------------------
# 드럼 추가 함수
# -------------------------
def add_drums(audio, bpm=140):

    beat = int(60000 / bpm)

    output = audio

    for i in range(0, len(audio), beat * 2):

        # 킥
        if kick:
            output = output.overlay(
                kick - 1,
                position=i
            )

        # 두번째 킥
        if kick:
            output = output.overlay(
                kick - 4,
                position=i + beat
            )

        # 스네어
        if snare:
            output = output.overlay(
                snare - 6,
                position=i + beat * 2
            )

        # 하이햇
        if hihat:
            output = output.overlay(
                hihat - 13,
                position=i + beat // 2
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
# 스타일 선택
# -------------------------
style = st.selectbox(
    "Choose Remix Style",
    [
        "Nightcore",
        "EDM Remix",
        "Slowed Reverb",
        "Bass Boost"
    ]
)

# -------------------------
# 리믹스 시작
# -------------------------
if uploaded:

    st.audio(uploaded)

    if st.button("Generate Remix"):

        progress = st.progress(0)

        status = st.empty()

        for i in range(101):

            time.sleep(0.02)

            progress.progress(i)

            status.write(f"Generating Remix... {i}%")

        # -------------------------
        # 임시 저장
        # -------------------------
        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(uploaded.read())

            temp_path = tmp.name

        audio = AudioSegment.from_file(temp_path)

        # -------------------------
        # NIGHTCORE
        # -------------------------
        if style == "Nightcore":

            # 너무 빠르지 않게
            remixed = speedup(audio, 1.08)

            # 살짝 피치 상승
            remixed = remixed._spawn(
                remixed.raw_data,
                overrides={
                    "frame_rate":
                    int(remixed.frame_rate * 1.04)
                }
            ).set_frame_rate(audio.frame_rate)

            # 드럼 추가
            remixed = add_drums(remixed, bpm=145)

            # 전체 볼륨
            remixed = remixed + 2

        # -------------------------
        # EDM REMIX
        # -------------------------
        elif style == "EDM Remix":

            remixed = speedup(audio, 1.05)

            remixed = add_drums(remixed, bpm=128)

            remixed = remixed + 4

        # -------------------------
        # SLOWED REVERB
        # -------------------------
        elif style == "Slowed Reverb":

            remixed = audio._spawn(
                audio.raw_data,
                overrides={
                    "frame_rate":
                    int(audio.frame_rate * 0.85)
                }
            ).set_frame_rate(audio.frame_rate)

            echo = remixed - 10

            remixed = remixed.overlay(
                echo,
                position=150
            )

        # -------------------------
        # BASS BOOST
        # -------------------------
        else:

            remixed = audio + 6

        # -------------------------
        # 저장
        # -------------------------
        output_path = "pacoel_remix.mp3"

        remixed.export(
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
