import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI


# -------------------------------------------------
# STREAMLIT + OPENAI SETUP
# -------------------------------------------------

# Get API key (Streamlit Cloud uses st.secrets; local can use env var)
api_key = ""
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.environ.get("OPENAI_API_KEY", "")

if not api_key:
    st.error("OpenAI API key not found. Add it in Streamlit Secrets or set OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=api_key)

# OpenAI per transcription request limit (~25MB)
OPENAI_MAX_BYTES = 25 * 1024 * 1024


# -------------------------------------------------
# FFMPEG HELPERS (convert + chunk)
# -------------------------------------------------

def run_ffmpeg(args: list[str]) -> None:
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[:2000])


def convert_to_mp3(input_path: str, out_mp3: str, bitrate_kbps: int = 48) -> None:
    """Convert any audio/video to a speech-optimized mp3 (smaller size)."""
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", f"{bitrate_kbps}k",
        out_mp3
    ])


def chunk_audio(input_mp3: str, out_dir: str, chunk_seconds: int) -> list[str]:
    """Split audio into chunks so each chunk stays under OpenAI's 25MB limit."""
    out_pattern = str(Path(out_dir) / "chunk_%03d.mp3")
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", input_mp3,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        out_pattern
    ])
    return sorted(str(p) for p in Path(out_dir).glob("chunk_*.mp3"))


# -------------------------------------------------
# UI
# -------------------------------------------------

st.set_page_config(page_title="AI Transcriber & Summarizer", layout="centered")

st.title("🎙️ AI Audio / Video Transcriber")
st.write(
    "Upload an audio or video file. The app will transcribe it using OpenAI "
    "and then generate a clean summary."
)

uploaded_file = st.file_uploader("Upload audio or video", type=None, key="main_uploader")

chunk_minutes = st.slider("Chunk size (minutes)", 5, 20, 10, 1)
bitrate_kbps = st.selectbox("Compression bitrate (kbps)", [32, 48, 64], index=1)

st.write("Max upload size (MB):", st.get_option("server.maxUploadSize"))
st.write("File selected:", uploaded_file.name if uploaded_file else None)

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"Uploaded file size: {file_size_mb:.2f} MB")

    if st.button("🚀 Transcribe & Summarize"):
        with st.spinner("Preparing file (convert + chunk)…"):
            with tempfile.TemporaryDirectory() as workdir:
                # Save upload to disk
                suffix = Path(uploaded_file.name).suffix or ".bin"
                in_path = str(Path(workdir) / f"input{suffix}")
                with open(in_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Convert to compressed mp3
                mp3_path = str(Path(workdir) / "speech.mp3")
                convert_to_mp3(in_path, mp3_path, bitrate_kbps=int(bitrate_kbps))

                # Chunk
                chunks = chunk_audio(mp3_path, workdir, chunk_seconds=chunk_minutes * 60)

                st.write(f"Transcribing {len(chunks)} chunk(s)…")
                prog = st.progress(0.0)

                transcript_parts: list[str] = []

                for i, chunk_path in enumerate(chunks, start=1):
                    # Safety check: each chunk must be <= 25MB for OpenAI
                    if os.path.getsize(chunk_path) > OPENAI_MAX_BYTES:
                        st.error(
                            "A chunk is still over 25MB. "
                            "Lower the bitrate or reduce the chunk size."
                        )
                        st.stop()

                    with open(chunk_path, "rb") as audio_file:
                        try:
                            transcription = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                            )
                        except Exception as e:
                            st.error(f"OpenAI transcription failed: {type(e).__name__}")
                            st.exception(e)
                            st.stop()

                    transcript_parts.append(transcription.text)
                    prog.progress(i / len(chunks))

                transcript_text = "\n\n".join(transcript_parts)

        st.success("Transcription complete!")

        # -------------------------------------------------
        # SUMMARY (GPT)
        # -------------------------------------------------
        with st.spinner("Generating summary…"):
            prompt = (
                "Summarize the following transcript.\n\n"
                "Return:\n"
                "1) A short title\n"
                "2) 5 key bullet points\n"
                "3) A short paragraph summary\n\n"
                f"TRANSCRIPT:\n{transcript_text}"
            )

            try:
                summary_response = client.responses.create(
                    model="gpt-4o-mini",
                    input=prompt,
                )
                summary_text = summary_response.output_text
            except Exception as e:
                st.error(f"OpenAI summarization failed: {type(e).__name__}")
                st.exception(e)
                st.stop()

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------
        st.subheader("📝 Summary")
        st.write(summary_text)

        st.subheader("📄 Full Transcript")
        st.text_area("Transcript", transcript_text, height=350)

        st.download_button("Download Summary (.txt)", summary_text, file_name="summary.txt")
        st.download_button("Download Transcript (.txt)", transcript_text, file_name="transcript.txt")
