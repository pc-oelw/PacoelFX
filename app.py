import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup, low_pass_filter, high_pass_filter
import tempfile
import os

st.set_page_config(
    page_title="Pacoel Wave",
    page_icon="🎵"
)

st.title("🎵 Pacoel Wave")
st.write("Automatic Remix Generator")

# -----------------------------
# 드럼 사운드 로드
# -----------------------------
def load_sound(path):
    if os.path.exists(path):
        return AudioSegment.from_file(path)
    return None

kick = load_sound("sounds/kick.wav")
snare = load_sound("sounds/snare.wav")
hihat = load_sound("sounds/hihat.wav")

# -----------------------------
# 드럼 추가 함수
# -----------------------------
def add_drums(audio, bpm=140):

    beat_interval = int(60000 / bpm)

    output = audio

    for i in range(0, len(audio), beat_interval):

        # 킥
        if kick:
            output = output.overlay(kick - 2, position=i)

        # 스네어
        if snare:
            output = output.overlay(
                snare - 4,
                position=i + beat_interval // 2
            )

        # 하이햇
        if hihat:
            output = output.overlay(
                hihat - 10,
                position=i + beat_interval // 4
            )

    return output

# -----------------------------
# Bass Boost
# -----------------------------
def bass_boost(audio):

    bass = low_pass_filter(audio, 120) + 8

    return audio.overlay(bass)

# -----------------------------
# Reverb 느낌
# -----------------------------
def fake_reverb(audio):

    echo = audio - 10

    combined = audio.overlay(echo, position=120)

    return combined

# -----------------------------
# Beat Drop
# -----------------------------
def beat_drop(audio):

    split = int(len(audio) * 0.7)

    first = audio[:split] - 2

    drop = audio[split:] + 6

    return first.append(drop, crossfade=300)

# -----------------------------
# 업로드
# -----------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

if uploaded:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        temp_path = tmp.name

    audio = AudioSegment.from_file(temp_path)

    st.audio(uploaded)

    remix_type = st.selectbox(
        "Choose Style",
        [
            "Nightcore",
            "Slowed Reverb",
            "EDM Remix",
            "Bass Boost"
        ]
    )

    if st.button("Generate Remix"):

        # -----------------------------
        # NIGHTCORE
        # -----------------------------
        if remix_type == "Nightcore":

            remixed = speedup(audio, 1.25)

            remixed = remixed._spawn(
                remixed.raw_data,
                overrides={
                    "frame_rate": int(remixed.frame_rate * 1.12)
                }
            ).set_frame_rate(audio.frame_rate)

            remixed = add_drums(remixed, bpm=160)

            remixed = high_pass_filter(remixed, 100)

        # -----------------------------
        # SLOWED REVERB
        # -----------------------------
        elif remix_type == "Slowed Reverb":

            remixed = audio._spawn(
                audio.raw_data,
                overrides={
                    "frame_rate": int(audio.frame_rate * 0.82)
                }
            ).set_frame_rate(audio.frame_rate)

            remixed = fake_reverb(remixed)

            remixed = bass_boost(remixed)

        # -----------------------------
        # EDM REMIX
        # -----------------------------
        elif remix_type == "EDM Remix":

            remixed = speedup(audio, 1.08)

            remixed = add_drums(remixed, bpm=140)

            remixed = bass_boost(remixed)

            remixed = beat_drop(remixed)

        # -----------------------------
        # BASS BOOST
        # -----------------------------
        elif remix_type == "Bass Boost":

            remixed = bass_boost(audio)

        # -----------------------------
        # 저장
        # -----------------------------
        output_path = "remixed.mp3"

        remixed.export(
            output_path,
            format="mp3"
        )

        st.success("Remix Complete!")

        st.audio(output_path)

        with open(output_path, "rb") as f:

            st.download_button(
                "Download Remix",
                f,
                file_name="pacoel_remix.mp3"
            )
