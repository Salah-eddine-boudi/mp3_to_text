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
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

# Configuration de la vitesse et des timestamps
CHUNK_LENGTH_S = 5               # Durée de chaque bloc
STRIDE_LENGTH_S = (1, 1)         # Recouvrement avant/après
RETURN_TIMESTAMPS = "word"      # Retourner timestamps par mot

# Mapping des langues
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

# Dossier de sortie
TEXT_OUTPUT_DIR = r"D:\stage 20242025\lake\mp3_transcipt"
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

# Création de la pipeline ASR avec timestamps
def get_pipeline_for_language(lang: str):
    model_id = LANG_MODEL_MAP.get(lang.lower(), DEFAULT_MODEL)
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S,
        return_timestamps=RETURN_TIMESTAMPS
    )

# Transcrire audio, extraire phrase par phrase avec timing et stocker
def transcrire_et_stocker(path_audio: str, doc_id: str, lang: str = 'en'):
    asr_pipe = get_pipeline_for_language(lang)
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
            # Appel ASR avec timestamps
            result = asr_pipe(block)
            chunks = result.get("chunks", [])
            for chunk in chunks:
                start_ms = chunk['timestamp'][0]
                text = chunk['text'].strip()
                if text:
                    ts = format_timestamp(start_ms)
                    lines.append(f"[{ts}] {text}")
            print(f"Bloc {i+1} ({lang}) traité avec {len(chunks)} segments")

    transcript_full = "\n".join(lines)

    # Sauvegarde dans le fichier texte
    base = os.path.splitext(os.path.basename(path_audio))[0]
    txt_path = os.path.join(TEXT_OUTPUT_DIR, f"{base}_{lang}.txt")
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(transcript_full)
    print(f"> Transcript ({lang}) écrit dans {txt_path}")

    # Stockage en base MongoDB
    col.update_one(
        {"_id": doc_id},
        {"$set": {"transcript": transcript_full,
                   "asr_model": LANG_MODEL_MAP.get(lang.lower(), DEFAULT_MODEL),
                   "language": lang}},
        upsert=True
    )
    print(f"> Document Mongo mis à jour pour {doc_id}, langue={lang}")

# Traitement en script direct
if __name__ == "__main__":
    folder = r"D:\stage 20242025\lake\mp3"
    for fn in os.listdir(folder):
        if fn.lower().endswith((".wav", ".mp3")):
            doc_id = os.path.splitext(fn)[0]
            path = os.path.join(folder, fn)
            for lang in LANG_MODEL_MAP.keys():
                transcrire_et_stocker(path, doc_id, lang)

# main.py inchangé, accepte toujours 'language' en JSON

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
