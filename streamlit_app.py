import os
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI

# -------------------------------------------------
# STREAMLIT + OPENAI SETUP
# -------------------------------------------------

# Load OpenAI key from Streamlit Secrets (Cloud-safe)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# OpenAI practical audio upload limit (~25MB)
MAX_AUDIO_BYTES = 25 * 1024 * 1024

st.set_page_config(
    page_title="AI Transcriber & Summarizer",
    layout="centered"
)

st.title("🎙️ AI Audio / Video Transcriber")
st.write(
    "Upload an audio or video file. The app will transcribe it using OpenAI "
    "and then generate a clean summary."
)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

uploaded_file = st.file_uploader("Upload audio or video", type=None, key="main_uploader")

st.write("Max upload size (MB):", st.get_option("server.maxUploadSize"))
st.write("File selected:", uploaded_file.name if uploaded_file else None)

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"Uploaded file size: {file_size_mb:.2f} MB")

    if uploaded_file.size > MAX_AUDIO_BYTES:
        st.error(
            "This file is larger than OpenAI's 25MB transcription limit.\n\n"
            "For now, please upload a shorter recording or a compressed audio file."
        )
        st.stop()

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please add it in Streamlit Secrets.")
        st.stop()

    if st.button("🚀 Transcribe & Summarize"):
        with st.spinner("Transcribing audio…"):
            # Save uploaded file to a temporary file
            suffix = Path(uploaded_file.name).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            try:
                # -------------------------------------------------
                # TRANSCRIPTION (Whisper)
                # -------------------------------------------------
                with open(temp_path, "rb") as audio:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio
                    )

                transcript_text = transcription.text

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

                    summary_response = client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt
                    )

                    summary_text = summary_response.output_text

                # -------------------------------------------------
                # DISPLAY RESULTS
                # -------------------------------------------------
                st.subheader("📝 Summary")
                st.write(summary_text)

                st.subheader("📄 Full Transcript")
                st.text_area(
                    "Transcript",
                    transcript_text,
                    height=350
                )

                # Download buttons
                st.download_button(
                    "Download Summary (.txt)",
                    summary_text,
                    file_name="summary.txt"
                )

                st.download_button(
                    "Download Transcript (.txt)",
                    transcript_text,
                    file_name="transcript.txt"
                )

            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
