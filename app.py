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
    layout="wide",
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

st.markdown("""
<div class="info-box">

Upload a song and Pacoel Wave will analyze the BPM,
preserve the original intro, build up the energy,
and create a speedcore-inspired drop with drums.

</div>
""", unsafe_allow_html=True)

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
# 안전한 speedup 함수
# -------------------------
def safe_speedup(audio, playback_speed):
    if playback_speed <= 1.01:
        return audio

    try:
        return speedup(
            audio,
            playback_speed=playback_speed,
            chunk_size=120,
            crossfade=25
        )
    except Exception:
        return audio


# -------------------------
# BPM 분석 함수
# -------------------------
def analyze_bpm(file_path):
    y, sr = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    tempo, beats = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    try:
        bpm = float(tempo[0])
    except Exception:
        bpm = float(tempo)

    if bpm <= 0 or np.isnan(bpm):
        bpm = 140

    beat_times = librosa.frames_to_time(
        beats,
        sr=sr
    )

    return int(bpm), beat_times


# -------------------------
# 드럼 추가 함수
# -------------------------
def add_drums(audio, bpm, section="build"):
    output = audio

    bpm = max(80, min(int(bpm), 260))
    beat_ms = int(60000 / bpm)

    if section == "intro":
        return output

    if section == "build":
        for pos in range(0, len(audio), beat_ms):
            if kick and pos % (beat_ms * 2) == 0:
                output = output.overlay(
                    kick - 8,
                    position=pos
                )

            if hihat:
                output = output.overlay(
                    hihat - 17,
                    position=pos + beat_ms // 2
                )

    elif section == "drop":
        # 스피드코어 느낌: 킥 밀도를 높임
        interval = max(115, beat_ms // 2)

        for pos in range(0, len(audio), interval):
            if kick:
                output = output.overlay(
                    kick,
                    position=pos
                )

            if hihat and pos % (interval * 2) == 0:
                output = output.overlay(
                    hihat - 11,
                    position=pos + 30
                )

            if snare and pos % (interval * 4) == 0:
                output = output.overlay(
                    snare - 6,
                    position=pos + 70
                )

    return output


# -------------------------
# 간단한 빌드업 효과
# -------------------------
def create_buildup(audio):
    length = len(audio)

    if length < 3000:
        return audio

    first = audio[:length // 2]
    second = audio[length // 2:] + 2

    return first.append(
        second,
        crossfade=200
    )


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
            "Uploading audio...",
            "Analyzing BPM...",
            "Preserving intro...",
            "Creating build-up...",
            "Generating speedcore drop...",
            "Adding drums...",
            "Finalizing remix..."
        ]

        for i in range(101):
            time.sleep(0.015)
            progress.progress(i)

            if i < 10:
                status.write(steps[0])
            elif i < 25:
                status.write(steps[1])
            elif i < 40:
                status.write(steps[2])
            elif i < 60:
                status.write(steps[3])
            elif i < 80:
                status.write(steps[4])
            elif i < 95:
                status.write(steps[5])
            else:
                status.write(steps[6])

        # -------------------------
        # 임시 파일 저장
        # -------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded.read())
            temp_path = tmp.name

        # -------------------------
        # BPM 분석
        # -------------------------
        bpm, beat_times = analyze_bpm(temp_path)
        st.write(f"Detected BPM: {bpm}")

        # -------------------------
        # 오디오 로드
        # -------------------------
        audio = AudioSegment.from_file(temp_path)

        total = len(audio)

        # 너무 짧은 파일 방지
        if total < 10000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        # -------------------------
        # 구조 분리
        # 처음은 최대한 원곡 유지
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
        # BPM별 속도 결정
        # -------------------------
        if bpm < 95:
            build_speed = 1.10
            drop_speed = 1.45
            drop_drum_bpm = 220

        elif bpm < 125:
            build_speed = 1.07
            drop_speed = 1.35
            drop_drum_bpm = 230

        elif bpm < 155:
            build_speed = 1.04
            drop_speed = 1.25
            drop_drum_bpm = 240

        else:
            build_speed = 1.02
            drop_speed = 1.15
            drop_drum_bpm = 250

        # -------------------------
        # 인트로 처리
        # 원곡 인식 가능하게 거의 유지
        # -------------------------
        intro = intro.fade_in(700)

        # -------------------------
        # 빌드업 처리
        # 살짝만 빠르게
        # -------------------------
        build = safe_speedup(
            build,
            build_speed
        )

        build = create_buildup(build)
        build = build + 1

        build = add_drums(
            build,
            bpm=int(bpm * build_speed),
            section="build"
        )

        # -------------------------
        # 드랍 처리
        # 여기서만 스피드코어 느낌
        # -------------------------
        drop = safe_speedup(
            drop,
            drop_speed
        )

        drop = drop + 4

        drop = add_drums(
            drop,
            bpm=drop_drum_bpm,
            section="drop"
        )

        # -------------------------
        # 드랍 직전 정적
        # -------------------------
        pause = AudioSegment.silent(
            duration=220
        )

        # -------------------------
        # 합치기
        # -------------------------
        remix = intro.append(
            build,
            crossfade=350
        )

        remix = remix.append(
            pause,
            crossfade=10
        )

        remix = remix.append(
            drop,
            crossfade=140
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

        st.success("Remix Complete!")

        st.audio(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "⬇ Download Remix",
                f,
                file_name="pacoel_speedcore_remix.mp3"
            )
