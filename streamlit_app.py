import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from math import floor

import streamlit as st
from openai import OpenAI

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload
from googleapiclient.errors import HttpError


# =================================================
# CONFIG
# =================================================
OPENAI_MAX_BYTES = 25 * 1024 * 1024  # 25MB per OpenAI transcription request


# =================================================
# OPENAI SETUP
# =================================================
api_key = ""
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.environ.get("OPENAI_API_KEY", "")

if not api_key:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)


# =================================================
# GOOGLE DRIVE HELPERS
# =================================================


def get_drive_service():
    # Expects Streamlit secrets:
    # [google_service_account]
    # type="service_account"
    # ...
    sa_info = dict(st.secrets["google_service_account"])

    # If your org policy blocks drive.file, switch to:
    # scopes=["https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)


def create_drive_folder(folder_name: str, parent_folder_id: str):
    service = get_drive_service()
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    created = (
        service.files()
        .create(
            body=metadata,
            fields="id, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )
    return created["id"], created.get("webViewLink")


def upload_text_to_drive(filename: str, content: str, folder_id: str):
    service = get_drive_service()
    media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/plain")

    file_metadata = {"name": filename, "parents": [folder_id]}

    created = (
        service.files()
        .create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink",
            supportsAllDrives=True,  # harmless for normal Drive, helpful for Shared Drives
        )
        .execute()
    )

    return created["id"], created.get("webViewLink")


def drive_debug_error(e: Exception) -> None:
    """
    Display the real Drive API error payload so you can fix permissions quickly.
    """
    if isinstance(e, HttpError):
        st.error("Google Drive API error (details below).")
        try:
            st.code(e.content.decode("utf-8"))
        except Exception:
            st.code(str(e))
    else:
        st.error(f"Google Drive upload failed: {type(e).__name__}")
        st.exception(e)


# =================================================
# FFMPEG HELPERS
# =================================================
def run_ffmpeg(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[:2000])


def convert_to_mp3(input_path, out_mp3, bitrate_kbps=48):
    # Speech-friendly compression: mono + 16kHz + low bitrate
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            f"{bitrate_kbps}k",
            out_mp3,
        ]
    )


def chunk_audio(input_mp3, out_dir, chunk_seconds):
    out_pattern = str(Path(out_dir) / "chunk_%03d.mp3")
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_mp3,
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-c",
            "copy",
            out_pattern,
        ]
    )
    return sorted(str(p) for p in Path(out_dir).glob("chunk_*.mp3"))


# =================================================
# UI
# =================================================
def format_timestamp(seconds: float) -> str:
    minutes = floor(seconds // 60)
    secs = floor(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


st.set_page_config(page_title="AI Transcriber & Summarizer", layout="centered")

st.title("🎙️ AI Audio / Video Transcriber")
st.write(
    "Upload an audio or video file. The app will transcribe it and generate a summary."
)

# Optional: confirm ffmpeg exists (helpful for local/Codespaces)
st.caption(f"ffmpeg detected: {shutil.which('ffmpeg') is not None}")

uploaded_file = st.file_uploader("Upload audio or video", type=None)
default_title = Path(uploaded_file.name).stem if uploaded_file else ""
doc_title = st.text_input("Recording / document title", value=default_title)
chunk_minutes = st.slider("Chunk size (minutes)", 5, 20, 10)
bitrate_kbps = st.selectbox("Compression bitrate (kbps)", [32, 48, 64], index=1)
save_to_drive = st.checkbox("Save transcript + summary to Google Drive", value=True)

# Drive debug panel (super useful)
with st.expander("Google Drive: debug / test access", expanded=False):
    st.write(
        "Secrets present:",
        {
            "GDRIVE_FOLDER_ID": "GDRIVE_FOLDER_ID" in st.secrets,
            "google_service_account": "google_service_account" in st.secrets,
        },
    )

    if st.button("Test Drive access to folder"):
        try:
            folder_id = st.secrets["GDRIVE_FOLDER_ID"]
            service = get_drive_service()
            meta = (
                service.files()
                .get(
                    fileId=folder_id,
                    fields="id,name,mimeType",
                    supportsAllDrives=True,
                )
                .execute()
            )
            st.success(
                f"✅ Service account can access folder: {meta['name']} ({meta['mimeType']})"
            )
            st.info(
                "If uploads still fail, it’s usually a permissions/scope policy on file creation."
            )
        except Exception as e:
            drive_debug_error(e)
            st.info(
                "Most common fix: share the Drive folder with your service account email "
                "(the client_email in your secrets) as Editor."
            )

if uploaded_file and st.button("🚀 Transcribe & Summarize"):
    # -----------------------------
    # TRANSCRIBE (convert + chunk)
    # -----------------------------
    with st.spinner("Preparing and chunking audio…"):
        with tempfile.TemporaryDirectory() as workdir:
            suffix = Path(uploaded_file.name).suffix or ".bin"
            input_path = str(Path(workdir) / f"input{suffix}")
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            mp3_path = str(Path(workdir) / "audio.mp3")
            convert_to_mp3(input_path, mp3_path, int(bitrate_kbps))

            chunks = chunk_audio(mp3_path, workdir, chunk_minutes * 60)

            st.write(f"Transcribing {len(chunks)} chunk(s)…")
            prog = st.progress(0.0)
            transcript_parts = []
            timestamped_segments = []
            chunk_duration = chunk_minutes * 60

            for i, chunk in enumerate(chunks, start=1):
                if os.path.getsize(chunk) > OPENAI_MAX_BYTES:
                    st.error("A chunk exceeded 25MB. Reduce bitrate or chunk size.")
                    st.stop()

                with open(chunk, "rb") as audio:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )

                chunk_offset = (i - 1) * chunk_duration
                transcript_parts.append(transcription.text)

                segments = []
                if hasattr(transcription, "segments"):
                    segments = transcription.segments or []
                elif isinstance(transcription, dict):
                    segments = transcription.get("segments") or []

                for seg in segments:
                    # seg can be a dict or a pydantic TranscriptionSegment
                    start_val = getattr(seg, "start", None)
                    end_val = getattr(seg, "end", None)
                    text_val = getattr(seg, "text", None)

                    if start_val is None and isinstance(seg, dict):
                        start_val = seg.get("start", 0)
                    if end_val is None and isinstance(seg, dict):
                        end_val = seg.get("end", 0)
                    if text_val is None and isinstance(seg, dict):
                        text_val = seg.get("text", "")

                    start = (start_val or 0) + chunk_offset
                    end = (end_val or 0) + chunk_offset
                    text = (text_val or "").strip()
                    timestamped_segments.append(
                        {
                            "start": start,
                            "end": end,
                            "text": text,
                        }
                    )

                prog.progress(i / len(chunks))

            # Construct a timestamped transcript at mm:ss resolution
            timestamped_transcript_lines = [
                f"[{format_timestamp(seg['start'])} - {format_timestamp(seg['end'])}] {seg['text']}"
                for seg in timestamped_segments
            ]
            timestamped_transcript_text = (
                "\n".join(timestamped_transcript_lines)
                if timestamped_segments
                else "\n\n".join(transcript_parts)
            )

            # Speaker labeling via GPT using the timestamped segments
            speaker_transcript_text = timestamped_transcript_text
            if timestamped_transcript_text:
                speaker_prompt = (
                    "You will receive timestamped transcript segments. "
                    "Identify different speakers and assign them names like 'Speaker 1', 'Speaker 2', etc. "
                    "Keep chronological order, merge contiguous segments from the same inferred speaker, "
                    "and preserve start/end timestamps (using the earliest start and latest end for merged text). "
                    "Format each line as: [MM:SS - MM:SS] Speaker X: text."
                )
                speaker_response = client.responses.create(
                    model="gpt-4o-mini",
                    input=(
                        speaker_prompt + "\n\nSEGMENTS:\n" + timestamped_transcript_text
                    ),
                )
                speaker_transcript_text = speaker_response.output_text

            transcript_text = speaker_transcript_text
            st.session_state["timestamped_transcript_text"] = (
                timestamped_transcript_text
            )

    st.success("Transcription complete!")
    st.session_state["transcript_text"] = transcript_text

    # -----------------------------
    # SUMMARY
    # -----------------------------
    with st.spinner("Generating summary…"):
        prompt = (
            "Summarize the following transcript.\n\n"
            "Return:\n"
            "1) A short title\n"
            "2) 5 key bullet points\n"
            "3) A short paragraph summary\n\n"
            f"TRANSCRIPT:\n{transcript_text}"
        )

        summary_response = client.responses.create(model="gpt-4o-mini", input=prompt)
        summary_text = summary_response.output_text

    st.session_state["summary_text"] = summary_text

    # -----------------------------
    # GOOGLE DRIVE EXPORT (IN ITS OWN FOLDER)
    # -----------------------------
    if save_to_drive:
        try:
            root_folder_id = st.secrets["GDRIVE_FOLDER_ID"]
            base = doc_title or Path(uploaded_file.name).stem

            upload_folder_id, upload_folder_link = create_drive_folder(
                folder_name=base, parent_folder_id=root_folder_id
            )

            t_id, t_link = upload_text_to_drive(
                f"{base}_transcript.txt", transcript_text, upload_folder_id
            )
            s_id, s_link = upload_text_to_drive(
                f"{base}_summary.txt", summary_text, upload_folder_id
            )

            st.success("✅ Saved to Google Drive in its own folder!")
            if upload_folder_link:
                st.link_button("Open folder in Drive", upload_folder_link)
            if t_link:
                st.link_button("Open Transcript", t_link)
            if s_link:
                st.link_button("Open Summary", s_link)

        except Exception as e:
            drive_debug_error(e)
            st.stop()

    # -----------------------------
    # DISPLAY
    # -----------------------------
    st.subheader("📝 Summary")
    st.write(summary_text)

    st.subheader("📄 Full Transcript (with timestamps & speakers)")
    st.text_area("Transcript", transcript_text, height=350)

    st.download_button("Download Summary (.txt)", summary_text, "summary.txt")
    st.download_button("Download Transcript (.txt)", transcript_text, "transcript.txt")
