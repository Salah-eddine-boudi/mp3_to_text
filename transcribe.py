# transcribe.py
import os
import soundfile as sf
import torch
import torchaudio
from pymongo import MongoClient
from transformers import pipeline

def format_timestamp(ms: float) -> str:
    """Convert milliseconds to MM:SS format."""
    total_s = int(ms / 1000)
    m, s = divmod(total_s, 60)
    return f"{m:02d}:{s:02d}"

# Parameters for ASR speed and overlap
CHUNK_LENGTH_S = 5           # duration of each audio block in seconds
STRIDE_LENGTH_S = (1, 1)     # overlap before and after each block
RETURN_TIMESTAMPS = "word"  # return timestamps at word level

# Language model mapping
LANG_MODEL_MAP = {
    'ar': 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
    'en': 'facebook/wav2vec2-large-960h',
    'fr': 'jonatasgrosman/wav2vec2-large-xlsr-fr',
    'es': 'jonatasgrosman/wav2vec2-large-xlsr-es',
    'tzm': 'facebook/wav2vec2-large-xlsr-53',
    'darija': 'facebook/wav2vec2-large-xlsr-53'
}
DEFAULT_MODEL = 'facebook/wav2vec2-large-xlsr-53'

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["audioDB"]
col = db["audioFiles"]

# Output directory for transcripts
TEXT_OUTPUT_DIR = r"D:\stage 20242025\lake\mp3_transcipt"
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

# Create ASR pipeline per language
def get_pipeline_for_language(lang: str):
    model_id = LANG_MODEL_MAP.get(lang.lower(), DEFAULT_MODEL)
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S,
        return_timestamps=RETURN_TIMESTAMPS
    )

# Transcribe audio into 10-word lines with timing
def transcrire_et_stocker(path_audio: str, doc_id: str, lang: str = 'en'):
    asr_pipe = get_pipeline_for_language(lang)
    word_list = []  # list of (start_ms, word)

    # Read audio and process in blocks
    with sf.SoundFile(path_audio) as f:
        sr = f.samplerate
        block_size = sr * CHUNK_LENGTH_S
        for i, block in enumerate(f.blocks(blocksize=block_size, dtype="float32")):
            # Mono conversion
            if block.ndim > 1:
                block = block.mean(axis=1)
            # Resample if needed
            if sr != 16000:
                block = torchaudio.functional.resample(
                    torch.tensor(block).unsqueeze(0), sr, 16000
                ).squeeze(0).numpy()
            # Perform ASR inference
            result = asr_pipe(block)
            chunks = result.get("chunks", [])
            for chunk in chunks:
                ts_start = chunk['timestamp'][0]
                text = chunk['text'].strip()
                if text:
                    word_list.append((ts_start, text))
            print(f"Bloc {i+1} ({lang}) traité : {len(chunks)} mots")

    # Group into lines of 10 words
    lines = []
    for idx in range(0, len(word_list), 10):
        group = word_list[idx:idx+10]
        if not group:
            continue
        start_ms = group[0][0]
        ts = format_timestamp(start_ms)
        words = [w for (_, w) in group]
        lines.append(f"[{ts}] {' '.join(words)}")

    transcript_full = "\n".join(lines)

    # Write to text file
    base = os.path.splitext(os.path.basename(path_audio))[0]
    txt_path = os.path.join(TEXT_OUTPUT_DIR, f"{base}_{lang}.txt")
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(transcript_full)
    print(f"> Transcript écrit dans {txt_path}")

    # Update MongoDB document
    col.update_one(
        {"_id": doc_id},
        {"$set": {
            "transcript": transcript_full,
            "asr_model": LANG_MODEL_MAP.get(lang.lower(), DEFAULT_MODEL),
            "language": lang
        }},
        upsert=True
    )
    print(f"> Document Mongo mis à jour pour {doc_id}, langue={lang}")

# If run as script, process all files for each language
if __name__ == "__main__":
    folder = r"D:\stage 20242025\lake\mp3"
    for fn in os.listdir(folder):
        if fn.lower().endswith((".wav", ".mp3")):
            doc_id = os.path.splitext(fn)[0]
            path_audio = os.path.join(folder, fn)
            for lang in LANG_MODEL_MAP.keys():
                transcrire_et_stocker(path_audio, doc_id, lang)

# main.py unchanged – accepts 'language' in JSON request

# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transcribe import transcrire_et_stocker

app = FastAPI(title="ASR Service")

class TranscriptionRequest(BaseModel):
    doc_id: str
    path_audio: str
    language: str = 'auto'

@app.post("/transcribe")
async def transcribe(req: TranscriptionRequest):
    try:
        transcrire_et_stocker(req.path_audio, req.doc_id, req.language)
        return {"status": "success", "doc_id": req.doc_id, "language": req.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
