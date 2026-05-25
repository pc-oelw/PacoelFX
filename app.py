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
    background-color: #e9e9ec;
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
    margin-bottom: 40px;
}

.box {
    background: white;
    padding: 30px;
    border-radius: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.stButton button {
    background-color: black;
    color: white;
    border-radius: 15px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stDownloadButton button {
    background-color: black;
    color: white;
    border-radius: 15px;
    height: 50px;
    width: 100%;
    font-size: 18px;
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
# 리믹스
# -------------------------
if uploaded:

    st.audio(uploaded)

    if st.button("Generate Remix"):

        progress = st.progress(0)

        status = st.empty()

        for i in range(101):

            time.sleep(0.02)

            progress.progress(i)

            status.write(f"Processing... {i}%")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(uploaded.read())

            temp_path = tmp.name

        audio = AudioSegment.from_file(temp_path)

        # -------------------------
        # NIGHTCORE
        # -------------------------
        if style == "Nightcore":

            remixed = speedup(audio, 1.25)

            remixed = remixed._spawn(
                remixed.raw_data,
                overrides={
                    "frame_rate":
                    int(remixed.frame_rate * 1.1)
                }
            ).set_frame_rate(audio.frame_rate)

        # -------------------------
        # EDM
        # -------------------------
        elif style == "EDM Remix":

            remixed = speedup(audio, 1.1)

            remixed = remixed + 5

        # -------------------------
        # SLOWED
        # -------------------------
        elif style == "Slowed Reverb":

            remixed = audio._spawn(
                audio.raw_data,
                overrides={
                    "frame_rate":
                    int(audio.frame_rate * 0.82)
                }
            ).set_frame_rate(audio.frame_rate)

        # -------------------------
        # BASS BOOST
        # -------------------------
        else:

            remixed = audio + 7

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
