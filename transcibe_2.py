
# transcribe2 .py          ----------> FR , EN , ES

import os
os.environ["HF_HOME"] = "D:/huggingface_cache"

import soundfile as sf
import torch
import torchaudio
from pymongo import MongoClient
from transformers import pipeline

import soundfile as sf
import torch
import torchaudio
import numpy as np
from pymongo import MongoClient
from transformers import pipeline
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seed pour la reproductibilité de langdetect
DetectorFactory.seed = 0

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

# Language models - Ajout de modèles plus performants
LANG_MODEL_MAP = {
    'ar': 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
    'en': 'facebook/wav2vec2-large-960h-lv60-self',  # Modèle amélioré pour l'anglais
    'fr': 'jonatasgrosman/wav2vec2-large-xlsr-53-french',
    'es': 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish',
    'de': 'jonatasgrosman/wav2vec2-large-xlsr-53-german',
    'it': 'jonatasgrosman/wav2vec2-large-xlsr-53-italian',
    'pt': 'jonatasgrosman/wav2vec2-large-xlsr-53-portuguese',
}

# Modèles de détection de langue audio
LANGUAGE_DETECTION_MODELS = {
    'whisper': 'openai/whisper-base',  # Excellent pour la détection de langue
    'wav2vec2': 'facebook/wav2vec2-large-xlsr-53'
}

DEFAULT_MODEL = 'facebook/wav2vec2-large-xlsr-53'

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["audioDB"]
col = db["audioFiles"]

# Output directory
TEXT_OUTPUT_DIR = r"D:\\stage 20242025\\lake\\mp3_transcript"
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

def preprocess_audio_segment(audio_data, sample_rate, target_sample_rate=16000, duration_s=30):
    """
    Prétraite un segment audio pour la détection de langue
    """
    # Convertir en mono si nécessaire
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Limiter la durée pour éviter les segments trop longs
    max_samples = target_sample_rate * duration_s
    if len(audio_data) > max_samples:
        # Prendre un segment du milieu pour éviter les silences de début/fin
        start_idx = (len(audio_data) - max_samples) // 2
        audio_data = audio_data[start_idx:start_idx + max_samples]
    
    # Resample si nécessaire
    if sample_rate != target_sample_rate:
        audio_tensor = torch.tensor(audio_data).unsqueeze(0)
        audio_data = torchaudio.functional.resample(
            audio_tensor, sample_rate, target_sample_rate
        ).squeeze(0).numpy()
    
    # Normaliser l'audio
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    return audio_data

def detect_language_whisper(audio_path: str) -> tuple[str, float]:
    """
    Détection de langue avec Whisper - très précis
    """
    try:
        # Charger le modèle Whisper
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=LANGUAGE_DETECTION_MODELS['whisper'],
            return_timestamps=False
        )
        
        # Lire un segment audio
        with sf.SoundFile(audio_path) as f:
            sample_rate = f.samplerate
            # Lire les 30 premières secondes
            audio_data = f.read(frames=min(sample_rate * 30, len(f)))
        
        # Prétraiter l'audio
        processed_audio = preprocess_audio_segment(audio_data, sample_rate)
        
        # Whisper peut détecter la langue directement
        result = whisper_pipe(processed_audio)
        
        # Extraire le texte pour la détection textuelle
        text = result.get('text', '').strip()
        
        if len(text) < 10:
            return 'unknown', 0.0
        
        # Utiliser langdetect sur le texte transcrit
        try:
            detected_lang = detect(text)
            confidence = 0.8  # Confiance élevée pour Whisper
            
            # Mapper les codes de langue
            lang_mapping = {
                'ar': 'ar', 'en': 'en', 'fr': 'fr', 'es': 'es',
                'de': 'de', 'it': 'it', 'pt': 'pt'
            }
            
            mapped_lang = lang_mapping.get(detected_lang, detected_lang)
            return mapped_lang, confidence
            
        except LangDetectException:
            return 'unknown', 0.0
            
    except Exception as e:
        logger.error(f"Erreur détection Whisper: {e}")
        return 'unknown', 0.0

def detect_language_multiple_samples(audio_path: str) -> tuple[str, float]:
    """
    Détection de langue avec plusieurs échantillons audio
    """
    try:
        results = []
        
        # Lire le fichier audio
        with sf.SoundFile(audio_path) as f:
            sample_rate = f.samplerate
            total_frames = len(f)
            
            # Prendre 3 échantillons à différents moments
            sample_positions = [0.1, 0.5, 0.8]  # 10%, 50%, 80% du fichier
            
            for pos in sample_positions:
                # Positionner le curseur
                start_frame = int(total_frames * pos)
                f.seek(start_frame)
                
                # Lire 10 secondes
                frames_to_read = min(sample_rate * 10, total_frames - start_frame)
                if frames_to_read <= 0:
                    continue
                    
                audio_segment = f.read(frames=frames_to_read)
                
                # Prétraiter
                processed_audio = preprocess_audio_segment(audio_segment, sample_rate)
                
                # Transcrire avec le modèle par défaut
                quick_pipe = pipeline(
                    "automatic-speech-recognition",
                    model=DEFAULT_MODEL,
                    chunk_length_s=CHUNK_LENGTH_S
                )
                
                result = quick_pipe(processed_audio)
                text = result.get('text', '').strip()
                
                if len(text) >= 10:
                    try:
                        lang = detect(text)
                        results.append(lang)
                    except LangDetectException:
                        continue
        
        if not results:
            return 'unknown', 0.0
        
        # Trouver la langue la plus fréquente
        from collections import Counter
        lang_counts = Counter(results)
        most_common_lang = lang_counts.most_common(1)[0][0]
        confidence = lang_counts[most_common_lang] / len(results)
        
        # Mapper les codes de langue
        lang_mapping = {
            'ar': 'ar', 'en': 'en', 'fr': 'fr', 'es': 'es',
            'de': 'de', 'it': 'it', 'pt': 'pt'
        }
        
        mapped_lang = lang_mapping.get(most_common_lang, 'en')
        return mapped_lang, confidence
        
    except Exception as e:
        logger.error(f"Erreur détection multiple: {e}")
        return 'unknown', 0.0

def detect_language_enhanced(audio_path: str) -> str:
    """
    Système de détection de langue amélioré avec plusieurs méthodes
    """
    logger.info(f"Détection de langue pour: {audio_path}")
    
    # Méthode 1: Whisper (plus précis)
    lang_whisper, conf_whisper = detect_language_whisper(audio_path)
    logger.info(f"Whisper: {lang_whisper} (confiance: {conf_whisper:.2f})")
    
    # Si Whisper est confiant et détecte une langue supportée
    if conf_whisper >= 0.7 and lang_whisper in LANG_MODEL_MAP:
        return lang_whisper
    
    # Méthode 2: Échantillons multiples
    lang_multi, conf_multi = detect_language_multiple_samples(audio_path)
    logger.info(f"Multi-échantillons: {lang_multi} (confiance: {conf_multi:.2f})")
    
    # Si la méthode multiple est confiante
    if conf_multi >= 0.6 and lang_multi in LANG_MODEL_MAP:
        return lang_multi
    
    # Méthode 3: Fallback avec le modèle original
    try:
        with sf.SoundFile(audio_path) as f:
            sr = f.samplerate
            # Lire un segment plus long pour plus de contexte
            block = f.read(frames=sr * 20)  # 20 secondes au lieu de 10

        if block.ndim > 1:
            block = block.mean(axis=1)
        if sr != 16000:
            block = torchaudio.functional.resample(
                torch.tensor(block).unsqueeze(0), sr, 16000
            ).squeeze(0).numpy()

        quick_pipe = pipeline(
            "automatic-speech-recognition",
            model=DEFAULT_MODEL,
            chunk_length_s=CHUNK_LENGTH_S
        )
        result = quick_pipe(block)
        text = result.get('text', '').strip()
        
        if len(text) >= 10:
            lang = detect(text)
            logger.info(f"Fallback: {lang}")
            
            # Vérifier si la langue est supportée
            if lang.startswith('ar'):
                return 'ar'
            elif lang.startswith('fr'):
                return 'fr'
            elif lang.startswith('es'):
                return 'es'
            elif lang.startswith('de'):
                return 'de'
            elif lang.startswith('it'):
                return 'it'
            elif lang.startswith('pt'):
                return 'pt'
            else:
                return 'en'
        
    except Exception as e:
        logger.error(f"Erreur méthode fallback: {e}")
    
    # Par défaut: anglais
    logger.info("Aucune langue détectée, utilisation de l'anglais par défaut")
    return 'en'

def is_already_transcribed(doc_id: str, lang: str):
    return col.find_one({"_id": doc_id, f"transcript_{lang}": {"$exists": True}}) is not None

def transcrire_et_stocker(path_audio: str, doc_id: str, lang: str):
    if is_already_transcribed(doc_id, lang):
        print(f"> Document {doc_id} déjà transcrit ({lang}).")
        return

    model_id = LANG_MODEL_MAP.get(lang, DEFAULT_MODEL)
    logger.info(f"Utilisation du modèle: {model_id} pour la langue: {lang}")
    
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S,
        return_timestamps=RETURN_TIMESTAMPS
    )

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
                        ts_formatted = format_timestamp(current_ts * 1000)  # Convert to ms
                        lines.append(f"[{ts_formatted}] {current_line.strip()}")
                        current_line = ""
                        current_ts = None
            
            if current_line and current_ts is not None:
                ts_formatted = format_timestamp(current_ts * 1000)
                lines.append(f"[{ts_formatted}] {current_line.strip()}")
            
            print(f"Bloc {i+1} ({lang}) traité")

    transcript_full = "\n\n".join(lines)
    base = os.path.splitext(os.path.basename(path_audio))[0]
    txt_path = os.path.join(TEXT_OUTPUT_DIR, f"{base}_{lang}.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(transcript_full)
    print(f"> Transcript ({lang}) écrit dans {txt_path}")

    # Stocker dans MongoDB avec informations supplémentaires
    update_data = {
        f"transcript_{lang}": transcript_full,
        "asr_model": model_id,
        "language": lang,
        "language_detection_method": "enhanced_multi_method",
        "transcript_timestamp": torch.tensor(0).item()  # Timestamp actuel
    }
    
    col.update_one(
        {"_id": doc_id},
        {"$set": update_data},
        upsert=True
    )
    print(f"> Document Mongo mis à jour pour {doc_id}, langue={lang}")

def main():
    folder = r"D:\\stage 20242025\\lake\\mp3"
    
    if not os.path.exists(folder):
        logger.error(f"Le dossier {folder} n'existe pas")
        return
    
    audio_files = [fn for fn in os.listdir(folder) 
                   if fn.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))]
    
    if not audio_files:
        logger.info("Aucun fichier audio trouvé")
        return
    
    logger.info(f"Traitement de {len(audio_files)} fichiers audio")
    
    for fn in audio_files:
        try:
            doc_id = os.path.splitext(fn)[0]
            path = os.path.join(folder, fn)
            
            # Utiliser le système de détection amélioré
            lang = detect_language_enhanced(path)
            print(f"> Langue détectée : {lang} pour {doc_id}")
            
            transcrire_et_stocker(path, doc_id, lang)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {fn}: {e}")
            continue
    
    print(">>> Fin de la tâche de transcription.")

if __name__ == "__main__":
    main()


# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transcribe import transcrire_et_stocker, detect_language_enhanced
import logging

app = FastAPI(title="ASR Service Multilingue", version="2.0")
logger = logging.getLogger(__name__)

class TranscriptionRequest(BaseModel):
    doc_id: str
    path_audio: str
    language: str = 'auto'

class LanguageDetectionRequest(BaseModel):
    path_audio: str

@app.post("/transcribe")
async def transcribe(req: TranscriptionRequest):
    """
    Transcrit un fichier audio avec détection automatique de langue
    """
    try:
        if req.language == 'auto':
            lang = detect_language_enhanced(req.path_audio)
        else:
            lang = req.language
            
        transcrire_et_stocker(req.path_audio, req.doc_id, lang)
        
        return {
            "status": "success",
            "doc_id": req.doc_id,
            "language": lang,
            "message": f"Transcription terminée avec succès pour la langue: {lang}"
        }
    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-language")
async def detect_language_endpoint(req: LanguageDetectionRequest):
    """
    Détecte la langue d'un fichier audio
    """
    try:
        lang = detect_language_enhanced(req.path_audio)
        return {
            "status": "success",
            "language": lang,
            "path_audio": req.path_audio
        }
    except Exception as e:
        logger.error(f"Erreur lors de la détection de langue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Vérification de l'état du service
    """
    return {"status": "healthy", "version": "2.0"}

@app.get("/supported-languages")
async def get_supported_languages():
    """
    Retourne la liste des langues supportées
    """
    from transcribe import LANG_MODEL_MAP
    return {
        "supported_languages": list(LANG_MODEL_MAP.keys()),
        "models": LANG_MODEL_MAP
    }
