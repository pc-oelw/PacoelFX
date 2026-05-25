import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup
import librosa
import numpy as np
import tempfile
import time
import os
import subprocess

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
    '<div class="sub-title">AI Stem Speedcore Remix Engine</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

Upload a song and Pacoel Wave will separate the track with Demucs AI,
preserve the original musical feeling, remove the original drums,
and create a speedcore-inspired remix with new drums.

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
# Demucs AI 분리 함수
# -------------------------
def separate_with_demucs(input_path):
    output_dir = tempfile.mkdtemp()

    command = [
        "python",
        "-m",
        "demucs",
        "-n",
        "htdemucs",
        "--out",
        output_dir,
        input_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    song_name = os.path.splitext(
        os.path.basename(input_path)
    )[0]

    stem_dir = os.path.join(
        output_dir,
        "htdemucs",
        song_name
    )

    vocals_path = os.path.join(stem_dir, "vocals.wav")
    drums_path = os.path.join(stem_dir, "drums.wav")
    bass_path = os.path.join(stem_dir, "bass.wav")
    other_path = os.path.join(stem_dir, "other.wav")

    if not os.path.exists(vocals_path):
        raise FileNotFoundError("vocals.wav was not created by Demucs.")

    if not os.path.exists(bass_path):
        raise FileNotFoundError("bass.wav was not created by Demucs.")

    if not os.path.exists(other_path):
        raise FileNotFoundError("other.wav was not created by Demucs.")

    return vocals_path, drums_path, bass_path, other_path


# -------------------------
# 드럼 제거된 반주 만들기
# -------------------------
def make_no_drums_audio(vocals_path, bass_path, other_path):
    vocals = AudioSegment.from_file(vocals_path)
    bass = AudioSegment.from_file(bass_path)
    other = AudioSegment.from_file(other_path)

    base = vocals.overlay(bass)
    base = base.overlay(other)

    return base


# -------------------------
# 드럼 추가 함수
# -------------------------
def add_drums(audio, bpm, section="build"):
    output = audio

    bpm = max(80, min(int(bpm), 280))
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
        interval = max(105, beat_ms // 2)

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
# 세션 상태
# -------------------------
if "remix_bytes" not in st.session_state:
    st.session_state.remix_bytes = None

if "remix_bpm" not in st.session_state:
    st.session_state.remix_bpm = None

if "remix_source" not in st.session_state:
    st.session_state.remix_source = None

if "demucs_used" not in st.session_state:
    st.session_state.demucs_used = None


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
    if st.session_state.remix_source != uploaded.name:
        st.session_state.remix_bytes = None
        st.session_state.remix_bpm = None
        st.session_state.demucs_used = None
        st.session_state.remix_source = uploaded.name

    st.audio(uploaded)

    if st.button("Generate AI Stem Remix"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing BPM...",
            "Separating stems with Demucs AI...",
            "Removing original drums...",
            "Preserving intro...",
            "Creating build-up...",
            "Generating speedcore drop...",
            "Adding new drums...",
            "Finalizing remix..."
        ]

        for i in range(101):
            time.sleep(0.015)
            progress.progress(i)

            if i < 8:
                status.write(steps[0])
            elif i < 18:
                status.write(steps[1])
            elif i < 38:
                status.write(steps[2])
            elif i < 48:
                status.write(steps[3])
            elif i < 58:
                status.write(steps[4])
            elif i < 70:
                status.write(steps[5])
            elif i < 84:
                status.write(steps[6])
            elif i < 95:
                status.write(steps[7])
            else:
                status.write(steps[8])

        # -------------------------
        # 임시 파일 저장
        # -------------------------
        file_ext = os.path.splitext(uploaded.name)[1]

        if file_ext.lower() not in [".mp3", ".wav"]:
            file_ext = ".mp3"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            uploaded.seek(0)
            tmp.write(uploaded.read())
            temp_path = tmp.name

        # -------------------------
        # BPM 분석
        # -------------------------
        bpm, beat_times = analyze_bpm(temp_path)

        # -------------------------
        # Demucs AI stem separation
        # -------------------------
        demucs_success = False

        try:
            status.write("Separating stems with Demucs AI...")

            vocals_path, drums_path, bass_path, other_path = separate_with_demucs(
                temp_path
            )

            audio = make_no_drums_audio(
                vocals_path,
                bass_path,
                other_path
            )

            demucs_success = True

        except Exception as e:
            st.warning(
                "Demucs AI separation failed, so Pacoel Wave used the original audio instead."
            )
            st.caption(str(e))

            audio = AudioSegment.from_file(temp_path)
            demucs_success = False

        # -------------------------
        # 오디오 길이 확인
        # -------------------------
        total = len(audio)

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
        output_path = "pacoel_ai_stem_speedcore_remix.mp3"

        remix.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.remix_bytes = f.read()

        st.session_state.remix_bpm = bpm
        st.session_state.demucs_used = demucs_success

    if st.session_state.remix_bytes:
        if st.session_state.remix_bpm is not None:
            st.write(f"Detected BPM: {st.session_state.remix_bpm}")

        if st.session_state.demucs_used:
            st.success("Demucs AI stem separation was used.")
        else:
            st.info("Original audio mode was used.")

        st.success("Remix Complete!")

        st.audio(
            st.session_state.remix_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download Remix",
            st.session_state.remix_bytes,
            file_name="pacoel_ai_stem_speedcore_remix.mp3",
            mime="audio/mpeg",
        )