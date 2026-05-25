import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup
import librosa
import numpy as np
import tempfile
import time
import random
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
    font-size: 62px;
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
    '<div class="sub-title">AI Chaos Speedcore Engine</div>',
    unsafe_allow_html=True
)

# -------------------------
# 설명
# -------------------------
st.markdown("""
<div class="info-box">

AI analyzes the uploaded song and dynamically generates
a chaotic speedcore-style remix with drops, BPM changes,
breakdowns, glitch cuts, and adaptive drum layering.

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
# 드럼 추가
# -------------------------
def add_chaos_drums(audio, intensity=1):

    if not kick:
        return audio

    output = audio

    interval = max(120, 320 - (intensity * 45))

    for pos in range(0, len(audio), interval):

        # 킥
        output = output.overlay(
            kick - (2 - intensity),
            position=pos
        )

        # 하이햇
        if hihat and random.random() > 0.2:

            output = output.overlay(
                hihat - 12,
                position=pos + random.randint(10, 70)
            )

        # 스네어
        if snare and random.random() > 0.5:

            output = output.overlay(
                snare - 6,
                position=pos + random.randint(60, 140)
            )

    return output

# -------------------------
# 글리치 컷
# -------------------------
def glitch(audio):

    chunks = []

    cursor = 0

    while cursor < len(audio):

        chunk_size = random.randint(80, 240)

        part = audio[cursor:cursor + chunk_size]

        mode = random.randint(0, 6)

        # 반복
        if mode == 0:

            chunks.append(part)
            chunks.append(part)

        # 리버스 느낌
        elif mode == 1:

            chunks.append(part.reverse())

        # 스킵
        elif mode == 2:

            pass

        # 정지 느낌
        elif mode == 3:

            chunks.append(AudioSegment.silent(duration=50))
            chunks.append(part)

        else:

            chunks.append(part)

        cursor += chunk_size

    final = AudioSegment.empty()

    for c in chunks:

        final += c

    return final

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

    if st.button("Generate AI Chaos Remix"):

        progress = st.progress(0)

        status = st.empty()

        steps = [
            "Analyzing BPM...",
            "Detecting Energy...",
            "Generating Build-up...",
            "Creating Drops...",
            "Injecting Chaos...",
            "Adding Speedcore Drums...",
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

        rms = librosa.feature.rms(y=y)[0]

        energy = np.mean(rms)

        # -------------------------
        # 오디오 로드
        # -------------------------
        audio = AudioSegment.from_file(temp_path)

        total = len(audio)

        # -------------------------
        # 섹션 분리
        # -------------------------
        intro_len = int(total * 0.18)

        buildup_len = int(total * 0.25)

        fake_drop_len = int(total * 0.12)

        intro = audio[:intro_len]

        buildup = audio[
            intro_len:intro_len + buildup_len
        ]

        fake_drop = audio[
            intro_len + buildup_len:
            intro_len + buildup_len + fake_drop_len
        ]

        drop = audio[
            intro_len + buildup_len + fake_drop_len:
        ]

        # -------------------------
        # BPM 계산
        # -------------------------
        if tempo < 100:

            buildup_speed = 1.18
            fake_speed = 1.45
            drop_speed = 1.80

        elif tempo < 140:

            buildup_speed = 1.12
            fake_speed = 1.35
            drop_speed = 1.65

        else:

            buildup_speed = 1.08
            fake_speed = 1.25
            drop_speed = 1.50

        # -------------------------
        # 인트로
        # -------------------------
        intro = intro.fade_in(1000)

        # -------------------------
        # 빌드업
        # -------------------------
        buildup = speedup(
            buildup,
            playback_speed=buildup_speed
        )

        buildup = buildup + 2

        # -------------------------
        # 페이크 드랍
        # -------------------------
        fake_drop = speedup(
            fake_drop,
            playback_speed=fake_speed
        )

        fake_drop = glitch(fake_drop)

        fake_drop = fake_drop + 4

        # -------------------------
        # 메인 드랍
        # -------------------------
        drop = speedup(
            drop,
            playback_speed=drop_speed
        )

        # 글리치
        drop = glitch(drop)

        # 드럼
        drop = add_chaos_drums(
            drop,
            intensity=4
        )

        # 볼륨
        drop = drop + 8

        # -------------------------
        # 브레이크다운
        # -------------------------
        breakdown = AudioSegment.silent(duration=250)

        # -------------------------
        # 합치기
        # -------------------------
        remix = intro.append(
            buildup,
            crossfade=300
        )

        remix = remix.append(
            breakdown,
            crossfade=20
        )

        remix = remix.append(
            fake_drop,
            crossfade=120
        )

        remix = remix.append(
            breakdown,
            crossfade=10
        )

        remix = remix.append(
            drop,
            crossfade=180
        )

        # -------------------------
        # 마스터
        # -------------------------
        remix = remix + 1

        # -------------------------
        # 저장
        # -------------------------
        output_path = "pacoel_chaos_speedcore.mp3"

        remix.export(
            output_path,
            format="mp3"
        )

        st.success("AI Chaos Remix Complete!")

        st.audio(output_path)

        with open(output_path, "rb") as f:

            st.download_button(
                "⬇ Download Remix",
                f,
                file_name="pacoel_chaos_speedcore.mp3"
            )
