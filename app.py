import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup, high_pass_filter
import librosa
import numpy as np
import tempfile
import time
import os
import subprocess

st.set_page_config(
    page_title="Pacoel Wave",
    page_icon="🎵",
    layout="wide"
)

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

st.markdown(
    '<div class="main-title">🎵 Pacoel Wave</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI J-Core Speedcore Remix Engine</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
Upload a song and Pacoel Wave will analyze the structure,
find an energy-based drop point, create a rising build-up,
then launch into a fast chaotic speedcore-style drop.
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
# 안전한 speedup
# -------------------------
def safe_speedup(audio, playback_speed):
    if playback_speed <= 1.01:
        return audio

    try:
        return speedup(
            audio,
            playback_speed=playback_speed,
            chunk_size=90,
            crossfade=20
        )
    except Exception:
        return audio


# -------------------------
# BPM + 비트 + 에너지 분석
# -------------------------
def analyze_audio(file_path):
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

    beat_ms = [
        int(t * 1000)
        for t in beat_times
    ]

    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr
    )

    return int(bpm), beat_ms, rms, rms_times


# -------------------------
# 에너지 기반 드랍 위치 찾기
# -------------------------
def find_drop_point_ms(total_ms, rms, rms_times, beat_ms):
    if len(rms) < 10:
        return int(total_ms * 0.62)

    # 너무 앞쪽/뒤쪽은 제외
    start_ratio = 0.38
    end_ratio = 0.78

    start_sec = (total_ms / 1000) * start_ratio
    end_sec = (total_ms / 1000) * end_ratio

    candidates = []

    for i in range(3, len(rms)):
        t = rms_times[i]

        if t < start_sec or t > end_sec:
            continue

        prev_energy = np.mean(rms[max(0, i - 3):i])
        now_energy = rms[i]

        # 에너지가 갑자기 커지는 지점
        rise = now_energy - prev_energy
        candidates.append((rise, t))

    if not candidates:
        drop_ms = int(total_ms * 0.62)
    else:
        candidates.sort(reverse=True)
        drop_ms = int(candidates[0][1] * 1000)

    # 가장 가까운 비트로 보정
    if beat_ms:
        nearby = [
            b for b in beat_ms
            if abs(b - drop_ms) < 1800
        ]

        if nearby:
            drop_ms = min(
                nearby,
                key=lambda x: abs(x - drop_ms)
            )

    drop_ms = max(
        int(total_ms * 0.42),
        min(drop_ms, int(total_ms * 0.78))
    )

    return drop_ms


# -------------------------
# 가까운 비트 찾기
# -------------------------
def nearest_beat_ms(target_ms, beat_ms):
    if not beat_ms:
        return target_ms

    nearby = [
        b for b in beat_ms
        if abs(b - target_ms) < 2000
    ]

    if not nearby:
        return target_ms

    return min(
        nearby,
        key=lambda x: abs(x - target_ms)
    )


# -------------------------
# Demucs
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
    bass_path = os.path.join(stem_dir, "bass.wav")
    other_path = os.path.join(stem_dir, "other.wav")

    if not os.path.exists(vocals_path):
        raise FileNotFoundError("vocals.wav was not created.")

    if not os.path.exists(bass_path):
        raise FileNotFoundError("bass.wav was not created.")

    if not os.path.exists(other_path):
        raise FileNotFoundError("other.wav was not created.")

    return vocals_path, bass_path, other_path


def make_no_drums_audio(vocals_path, bass_path, other_path):
    vocals = AudioSegment.from_file(vocals_path)
    bass = AudioSegment.from_file(bass_path)
    other = AudioSegment.from_file(other_path)

    base = vocals.overlay(bass)
    base = base.overlay(other)

    return base


# -------------------------
# 인트로 처리
# -------------------------
def process_intro(audio, bpm):
    intro = audio.fade_in(350)
    intro = high_pass_filter(intro, 35)

    # 초반도 너무 그대로가 아니게 약한 하이햇 추가
    if hihat:
        beat_ms = int(60000 / max(80, bpm))
        start = int(len(intro) * 0.55)
        end = len(intro) - 800

        for pos in range(start, max(start, end), beat_ms * 2):
            intro = intro.overlay(
                hihat - 21,
                position=pos
            )

    return intro


# -------------------------
# 빌드업: 점점 빨라지는 구조
# -------------------------
def process_buildup_ramp(audio, bpm):
    if len(audio) < 4000:
        return audio

    parts = 4
    part_len = len(audio) // parts

    speeds = [1.00, 1.04, 1.08, 1.13]
    result = AudioSegment.empty()

    for i in range(parts):
        chunk = audio[i * part_len:(i + 1) * part_len]

        chunk = safe_speedup(
            chunk,
            speeds[i]
        )

        # 뒤로 갈수록 볼륨 상승
        chunk = chunk + i

        result = result.append(
            chunk,
            crossfade=80 if len(result) > 0 else 0
        )

    beat_ms = int(60000 / max(90, bpm))
    end_limit = len(result) - 700

    for pos in range(0, max(0, end_limit), beat_ms):
        # 빌드업은 킥 약하게
        if kick and pos > len(result) * 0.45 and pos % (beat_ms * 2) == 0:
            result = result.overlay(
                kick - 9,
                position=pos
            )

        # 하이햇은 뒤로 갈수록 촘촘하게
        if hihat:
            if pos < len(result) * 0.55:
                interval_check = beat_ms * 2
            else:
                interval_check = beat_ms

            if pos % interval_check == 0:
                result = result.overlay(
                    hihat - 17,
                    position=pos + beat_ms // 2
                )

    return result


# -------------------------
# 드랍 전 효과
# -------------------------
def make_pre_drop():
    # 갑자기 멈추는 느낌
    return AudioSegment.silent(duration=260)


# -------------------------
# 스피드코어 드랍
# -------------------------
def process_speedcore_drop(audio, source_bpm):
    # 음악 자체는 너무 과하게 망가뜨리지 않고,
    # 킥 밀도와 드럼 BPM으로 스피드코어 느낌을 만듦
    if source_bpm < 100:
        music_speed = 1.18
        drum_bpm = 230
    elif source_bpm < 135:
        music_speed = 1.13
        drum_bpm = 240
    else:
        music_speed = 1.08
        drum_bpm = 250

    drop = safe_speedup(
        audio,
        music_speed
    )

    drop = drop + 3

    beat_ms = int(60000 / drum_bpm)

    # 200BPM 이상 느낌: 1/2박 킥
    kick_interval = max(95, beat_ms // 2)

    # 끝까지 드럼을 넣지 않고 페이드 전에 멈춤
    end_limit = len(drop) - 1700

    for pos in range(0, max(0, end_limit), kick_interval):
        if kick:
            drop = drop.overlay(
                kick + 1,
                position=pos
            )

        # 하이햇
        if hihat and pos % (kick_interval * 2) == 0:
            drop = drop.overlay(
                hihat - 10,
                position=pos + 25
            )

        # 스네어
        if snare and pos % (kick_interval * 4) == 0:
            drop = drop.overlay(
                snare - 6,
                position=pos + 60
            )

        # 중간중간 더블킥 느낌
        if kick and pos % (kick_interval * 8) == 0:
            extra_pos = pos + int(kick_interval * 0.55)

            if extra_pos < end_limit:
                drop = drop.overlay(
                    kick - 2,
                    position=extra_pos
                )

    drop = drop.fade_out(1300)

    return drop, drum_bpm


# -------------------------
# 세션 상태
# -------------------------
if "remix_bytes" not in st.session_state:
    st.session_state.remix_bytes = None

if "remix_bpm" not in st.session_state:
    st.session_state.remix_bpm = None

if "drop_bpm" not in st.session_state:
    st.session_state.drop_bpm = None

if "demucs_used" not in st.session_state:
    st.session_state.demucs_used = None

if "remix_source" not in st.session_state:
    st.session_state.remix_source = None


# -------------------------
# 업로드
# -------------------------
uploaded = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"]
)

if uploaded:
    if st.session_state.remix_source != uploaded.name:
        st.session_state.remix_bytes = None
        st.session_state.remix_bpm = None
        st.session_state.drop_bpm = None
        st.session_state.demucs_used = None
        st.session_state.remix_source = uploaded.name

    st.audio(uploaded)

    if st.button("Generate J-Core Speedcore Remix"):
        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing BPM and energy...",
            "Finding drop point...",
            "Separating stems with Demucs AI...",
            "Creating intro variation...",
            "Building tempo ramp...",
            "Preparing sudden drop...",
            "Generating speedcore drums...",
            "Finalizing remix..."
        ]

        for i in range(101):
            time.sleep(0.012)
            progress.progress(i)

            if i < 8:
                status.write(steps[0])
            elif i < 20:
                status.write(steps[1])
            elif i < 30:
                status.write(steps[2])
            elif i < 48:
                status.write(steps[3])
            elif i < 58:
                status.write(steps[4])
            elif i < 70:
                status.write(steps[5])
            elif i < 80:
                status.write(steps[6])
            elif i < 94:
                status.write(steps[7])
            else:
                status.write(steps[8])

        file_ext = os.path.splitext(uploaded.name)[1]

        if file_ext.lower() not in [".mp3", ".wav"]:
            file_ext = ".mp3"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            uploaded.seek(0)
            tmp.write(uploaded.read())
            temp_path = tmp.name

        bpm, beat_ms, rms, rms_times = analyze_audio(temp_path)

        original_audio = AudioSegment.from_file(temp_path)

        if len(original_audio) < 15000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        demucs_success = False

        try:
            status.write("Separating stems with Demucs AI...")

            vocals_path, bass_path, other_path = separate_with_demucs(
                temp_path
            )

            base_audio = make_no_drums_audio(
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

            base_audio = original_audio
            demucs_success = False

        total = len(base_audio)

        # 에너지 기반 드랍 지점
        drop_start = find_drop_point_ms(
            total,
            rms,
            rms_times,
            beat_ms
        )

        # 인트로 끝은 드랍 전 35~45% 지점 정도
        intro_target = int(drop_start * 0.45)

        intro_end = nearest_beat_ms(
            intro_target,
            beat_ms
        )

        intro_end = max(
            5000,
            min(intro_end, int(total * 0.35))
        )

        drop_start = max(
            intro_end + 6000,
            drop_start
        )

        drop_start = min(
            drop_start,
            int(total * 0.82)
        )

        intro = base_audio[:intro_end]
        build = base_audio[intro_end:drop_start]
        drop = base_audio[drop_start:]

        intro = process_intro(
            intro,
            bpm
        )

        build = process_buildup_ramp(
            build,
            bpm
        )

        pre_drop = make_pre_drop()

        drop, drop_bpm = process_speedcore_drop(
            drop,
            bpm
        )

        remix = intro.append(
            build,
            crossfade=220
        )

        remix = remix.append(
            pre_drop,
            crossfade=5
        )

        remix = remix.append(
            drop,
            crossfade=60
        )

        remix = remix.fade_out(1000)

        # 원곡보다 너무 길어지지 않게
        if len(remix) > len(original_audio):
            remix = remix[:len(original_audio)]

        output_path = "pacoel_jcore_speedcore_remix.mp3"

        remix.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.remix_bytes = f.read()

        st.session_state.remix_bpm = bpm
        st.session_state.drop_bpm = drop_bpm
        st.session_state.demucs_used = demucs_success

    if st.session_state.remix_bytes:
        st.write(f"Detected BPM: {st.session_state.remix_bpm}")
        st.write(f"Speedcore Drop BPM: {st.session_state.drop_bpm}")

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
            file_name="pacoel_jcore_speedcore_remix.mp3",
            mime="audio/mpeg",
        )