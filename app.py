import streamlit as st
from pydub import AudioSegment
from pydub.effects import speedup, high_pass_filter
import librosa
import numpy as np
import tempfile
import time
import os
import requests
import replicate

# -------------------------
# 페이지 설정
# -------------------------
st.set_page_config(
    page_title="Pacoel Wave",
    page_icon="🎵",
    layout="wide"
)

# -------------------------
# Replicate API 토큰
# -------------------------
try:
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
    REPLICATE_READY = True
except Exception:
    REPLICATE_READY = False

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
    '<div class="sub-title">Generative AI Speedcore Remix Engine</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

Upload a song. Pacoel Wave analyzes its BPM and structure,
then uses generative AI to create a new J-core / speedcore-style drop
and blends it into your track.

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
# BPM / 에너지 분석
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
# 드랍 위치 찾기
# -------------------------
def find_drop_point_ms(total_ms, rms, rms_times, beat_ms):
    if len(rms) < 10:
        return int(total_ms * 0.62)

    start_sec = (total_ms / 1000) * 0.35
    end_sec = (total_ms / 1000) * 0.78

    candidates = []

    for i in range(4, len(rms)):
        t = rms_times[i]

        if t < start_sec or t > end_sec:
            continue

        before = np.mean(rms[max(0, i - 4):i])
        now = rms[i]
        rise = now - before

        candidates.append((rise, t))

    if not candidates:
        drop_ms = int(total_ms * 0.62)
    else:
        candidates.sort(reverse=True)
        drop_ms = int(candidates[0][1] * 1000)

    if beat_ms:
        nearby = [
            b for b in beat_ms
            if abs(b - drop_ms) < 2000
        ]

        if nearby:
            drop_ms = min(
                nearby,
                key=lambda x: abs(x - drop_ms)
            )

    drop_ms = max(
        int(total_ms * 0.42),
        min(drop_ms, int(total_ms * 0.82))
    )

    return drop_ms


# -------------------------
# AI 프롬프트 만들기
# -------------------------
def make_ai_prompt(bpm):
    if bpm < 100:
        target_bpm = 230
    elif bpm < 135:
        target_bpm = 240
    else:
        target_bpm = 250

    prompt = (
        f"chaotic J-core speedcore drop, {target_bpm} BPM, "
        "aggressive distorted kicks, rapid drum fills, bright arcade synth leads, "
        "sudden tempo switch, rhythm game boss song energy, intense electronic drop, "
        "hyper energetic, clean mix, instrumental, no vocals"
    )

    return prompt, target_bpm


# -------------------------
# Replicate MusicGen 호출
# -------------------------
def generate_ai_music(prompt, duration=18):
    output = replicate.run(
        "meta/musicgen",
        input={
            "prompt": prompt,
            "duration": duration,
            "model_version": "stereo-large",
            "output_format": "mp3",
            "temperature": 1.05,
            "classifier_free_guidance": 4
        }
    )

    # Replicate 출력이 list일 수도 있고 string/url일 수도 있음
    if isinstance(output, list):
        output_url = output[0]
    else:
        output_url = output

    return str(output_url)


# -------------------------
# URL에서 AI 오디오 다운로드
# -------------------------
def download_ai_audio(url):
    response = requests.get(
        url,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError("Failed to download AI audio.")

    temp_ai = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    temp_ai.write(response.content)
    temp_ai.close()

    return temp_ai.name


# -------------------------
# 인트로 변형
# -------------------------
def process_intro(audio, bpm):
    intro = audio.fade_in(350)
    intro = high_pass_filter(intro, 35)

    # 초반도 약간 바뀌게 약한 하이햇과 킥 추가
    beat_ms = int(60000 / max(80, bpm))

    start = int(len(intro) * 0.35)
    end = len(intro) - 1000

    for pos in range(start, max(start, end), beat_ms * 2):
        if hihat:
            intro = intro.overlay(
                hihat - 21,
                position=pos + beat_ms // 2
            )

        if kick and pos > len(intro) * 0.65:
            intro = intro.overlay(
                kick - 12,
                position=pos
            )

    return intro


# -------------------------
# 빌드업 처리
# -------------------------
def process_buildup(audio, bpm):
    if len(audio) < 4000:
        return audio

    parts = 4
    part_len = len(audio) // parts

    speeds = [1.00, 1.03, 1.06, 1.10]
    result = AudioSegment.empty()

    for i in range(parts):
        chunk = audio[i * part_len:(i + 1) * part_len]
        chunk = safe_speedup(chunk, speeds[i])
        chunk = chunk + i

        if len(result) == 0:
            result += chunk
        else:
            result = result.append(
                chunk,
                crossfade=80
            )

    beat_ms = int(60000 / max(90, bpm))
    end_limit = len(result) - 700

    for pos in range(0, max(0, end_limit), beat_ms):
        if hihat:
            result = result.overlay(
                hihat - 17,
                position=pos + beat_ms // 2
            )

        if kick and pos > len(result) * 0.45 and pos % (beat_ms * 2) == 0:
            result = result.overlay(
                kick - 9,
                position=pos
            )

    return result


# -------------------------
# AI 드랍 후처리
# -------------------------
def process_ai_drop(ai_audio, target_bpm):
    drop = ai_audio

    # 너무 작거나 밋밋하면 살짝 강화
    drop = drop + 2

    beat_ms = int(60000 / target_bpm)
    kick_interval = max(95, beat_ms // 2)

    end_limit = len(drop) - 1500

    for pos in range(0, max(0, end_limit), kick_interval):
        if kick:
            drop = drop.overlay(
                kick + 1,
                position=pos
            )

        if hihat and pos % (kick_interval * 2) == 0:
            drop = drop.overlay(
                hihat - 10,
                position=pos + 25
            )

        if snare and pos % (kick_interval * 4) == 0:
            drop = drop.overlay(
                snare - 6,
                position=pos + 60
            )

    drop = drop.fade_in(80)
    drop = drop.fade_out(1200)

    return drop


# -------------------------
# 세션 상태
# -------------------------
if "remix_bytes" not in st.session_state:
    st.session_state.remix_bytes = None

if "remix_bpm" not in st.session_state:
    st.session_state.remix_bpm = None

if "drop_bpm" not in st.session_state:
    st.session_state.drop_bpm = None

if "ai_used" not in st.session_state:
    st.session_state.ai_used = None

if "remix_source" not in st.session_state:
    st.session_state.remix_source = None


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
        st.session_state.drop_bpm = None
        st.session_state.ai_used = None
        st.session_state.remix_source = uploaded.name

    st.audio(uploaded)

    if not REPLICATE_READY:
        st.warning(
            "REPLICATE_API_TOKEN is missing. Add it in Streamlit Secrets first."
        )

    if st.button("Generate Generative AI Remix"):
        if not REPLICATE_READY:
            st.error("Replicate API token is not set.")
            st.stop()

        progress = st.progress(0)
        status = st.empty()

        steps = [
            "Uploading audio...",
            "Analyzing BPM and energy...",
            "Finding drop point...",
            "Creating AI prompt...",
            "Generating AI speedcore drop...",
            "Downloading AI audio...",
            "Creating intro variation...",
            "Creating build-up...",
            "Blending AI drop...",
            "Finalizing remix..."
        ]

        for i in range(101):
            time.sleep(0.012)
            progress.progress(i)

            if i < 8:
                status.write(steps[0])
            elif i < 18:
                status.write(steps[1])
            elif i < 28:
                status.write(steps[2])
            elif i < 35:
                status.write(steps[3])
            elif i < 60:
                status.write(steps[4])
            elif i < 68:
                status.write(steps[5])
            elif i < 76:
                status.write(steps[6])
            elif i < 86:
                status.write(steps[7])
            elif i < 95:
                status.write(steps[8])
            else:
                status.write(steps[9])

        # -------------------------
        # 임시 저장
        # -------------------------
        file_ext = os.path.splitext(uploaded.name)[1]

        if file_ext.lower() not in [".mp3", ".wav"]:
            file_ext = ".mp3"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            uploaded.seek(0)
            tmp.write(uploaded.read())
            temp_path = tmp.name

        # -------------------------
        # 분석
        # -------------------------
        bpm, beat_ms, rms, rms_times = analyze_audio(temp_path)

        original_audio = AudioSegment.from_file(temp_path)

        if len(original_audio) < 15000:
            st.error("Audio is too short. Please upload a longer song.")
            st.stop()

        total = len(original_audio)

        drop_start = find_drop_point_ms(
            total,
            rms,
            rms_times,
            beat_ms
        )

        intro_target = int(drop_start * 0.45)
        intro_end = intro_target

        if beat_ms:
            nearby_intro = [
                b for b in beat_ms
                if abs(b - intro_target) < 2000
            ]

            if nearby_intro:
                intro_end = min(
                    nearby_intro,
                    key=lambda x: abs(x - intro_target)
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

        intro = original_audio[:intro_end]
        build = original_audio[intro_end:drop_start]

        # -------------------------
        # AI 생성
        # -------------------------
        prompt, target_bpm = make_ai_prompt(bpm)

        status.write("Generating AI speedcore drop...")

        try:
            ai_url = generate_ai_music(
                prompt,
                duration=18
            )

            status.write("Downloading AI audio...")

            ai_path = download_ai_audio(ai_url)

            ai_audio = AudioSegment.from_file(ai_path)

            ai_drop = process_ai_drop(
                ai_audio,
                target_bpm
            )

            ai_success = True

        except Exception as e:
            st.warning("AI generation failed. Using fallback speedcore drop.")
            st.caption(str(e))

            fallback = original_audio[drop_start:]
            fallback = safe_speedup(fallback, 1.18)
            ai_drop = process_ai_drop(
                fallback,
                target_bpm
            )

            ai_success = False

        # -------------------------
        # 인트로 / 빌드업
        # -------------------------
        intro = process_intro(
            intro,
            bpm
        )

        build = process_buildup(
            build,
            bpm
        )

        pre_drop = AudioSegment.silent(
            duration=250
        )

        # -------------------------
        # 합치기
        # -------------------------
        remix = intro.append(
            build,
            crossfade=220
        )

        remix = remix.append(
            pre_drop,
            crossfade=5
        )

        remix = remix.append(
            ai_drop,
            crossfade=80
        )

        remix = remix.fade_out(1000)

        output_path = "pacoel_generative_ai_speedcore_remix.mp3"

        remix.export(
            output_path,
            format="mp3"
        )

        with open(output_path, "rb") as f:
            st.session_state.remix_bytes = f.read()

        st.session_state.remix_bpm = bpm
        st.session_state.drop_bpm = target_bpm
        st.session_state.ai_used = ai_success

    if st.session_state.remix_bytes:
        st.write(f"Detected BPM: {st.session_state.remix_bpm}")
        st.write(f"AI Drop Target BPM: {st.session_state.drop_bpm}")

        if st.session_state.ai_used:
            st.success("Generative AI drop was used.")
        else:
            st.info("Fallback drop was used.")

        st.success("Remix Complete!")

        st.audio(
            st.session_state.remix_bytes,
            format="audio/mp3"
        )

        st.download_button(
            "⬇ Download Remix",
            st.session_state.remix_bytes,
            file_name="pacoel_generative_ai_speedcore_remix.mp3",
            mime="audio/mpeg",
        )