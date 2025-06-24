# transcribe.py
import os
import soundfile as sf
import torch
import torchaudio
from pymongo import MongoClient
from transformers import pipeline


def format_timestamp(ms: float) -> str:
    """Convert milliseconds to HH:MM:SS format."""
    s = int(ms / 1000)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ASR configuration
CHUNK_LENGTH_S = 10           # Longer chunk duration for fewer blocks
STRIDE_LENGTH_S = (1, 1)      # Overlap in seconds
RETURN_TIMESTAMPS = "word"   # Word-level timestamps

# Arabic language model
AR_MODEL = 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic'

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["audioDB"]
col = db["audioFiles"]

# Output directory
TEXT_OUTPUT_DIR = r"D:\stage 20242025\lake\mp3_transcript"
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

# ASR pipeline for Arabic
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=AR_MODEL,
    chunk_length_s=CHUNK_LENGTH_S,
    stride_length_s=STRIDE_LENGTH_S,
    return_timestamps=RETURN_TIMESTAMPS
)

# Check if transcription already exists
def is_already_transcribed(doc_id: str):
    return col.find_one({"_id": doc_id, "transcript_ar": {"$exists": True}}) is not None

# Transcribe and store Arabic audio with enhanced formatting
def transcrire_et_stocker_arabe(path_audio: str, doc_id: str):
    if is_already_transcribed(doc_id):
        print(f"> Document {doc_id} déjà transcrit.")
        return

    lines = []

    with sf.SoundFile(path_audio) as f:
        sr = f.samplerate
        block_size = sr * CHUNK_LENGTH_S
        for i, block in enumerate(f.blocks(blocksize=block_size, dtype="float32")):
            if block.ndim > 1:
                block = block.mean(axis=1)
            if sr != 16000:
                block = torchaudio.functional.resample(
                    torch.tensor(block).unsqueeze(0), sr, 16000
                ).squeeze(0).numpy()
            # ASR inference
            result = asr_pipe(block)
            current_line = ""
            current_ts = None
            for chunk in result.get("chunks", []):
                text = chunk['text'].strip()
                if text:
                    if current_ts is None:
                        current_ts = chunk['timestamp'][0]
                    current_line += " " + text
                    if len(current_line.split()) >= 10:
                        ts_formatted = format_timestamp(current_ts)
                        lines.append(f"[{ts_formatted}] {current_line.strip()}")
                        current_line = ""
                        current_ts = None
            if current_line:
                ts_formatted = format_timestamp(current_ts)
                lines.append(f"[{ts_formatted}] {current_line.strip()}")
            print(f"Bloc {i+1} (ar) traité")

    transcript_full = "\n\n".join(lines)  # Double line breaks for better readability

    # Save transcript to text file
    base = os.path.splitext(os.path.basename(path_audio))[0]
    txt_path = os.path.join(TEXT_OUTPUT_DIR, f"{base}_ar.txt")
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(transcript_full)
    print(f"> Transcript arabe écrit dans {txt_path}")

    # Update MongoDB
    col.update_one(
        {"_id": doc_id},
        {"$set": {"transcript_ar": transcript_full, "asr_model": AR_MODEL}},
        upsert=True
    )
    print(f"> Document Mongo mis à jour pour {doc_id} (arabe)")

# Main script entry point
def main():
    folder = r"D:\stage 20242025\lake\mp3"
    for fn in os.listdir(folder):
        if fn.lower().endswith((".wav", ".mp3")):
            doc_id = os.path.splitext(fn)[0]
            path = os.path.join(folder, fn)
            transcrire_et_stocker_arabe(path, doc_id)

if __name__ == "__main__":
    main()


# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transcribe import transcrire_et_stocker_arabe

app = FastAPI(title="ASR Service Arabe")

class TranscriptionRequest(BaseModel):
    doc_id: str
    path_audio: str

@app.post("/transcribe_ar")
async def transcribe_ar(req: TranscriptionRequest):
    try:
        transcrire_et_stocker_arabe(req.path_audio, req.doc_id)
        return {"status": "success", "doc_id": req.doc_id, "language": "ar"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
