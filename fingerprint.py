# fingerprint.py
from dejavu import Dejavu
import json
import os

# Charger le fichier de configuration MySQL
with open("dejavu.cnf") as f:
    config = json.load(f)

# Initialiser Dejavu avec la config
djv = Dejavu(config)

# Chemin du dossier contenant les fichiers audio à indexer
AUDIO_DIR = "audios"  # Change ça selon ton organisation

# Vérifier que le dossier existe
if not os.path.exists(AUDIO_DIR):
    print(f"Dossier '{AUDIO_DIR}' introuvable. Crée-le et mets-y tes fichiers audio.")
else:
    # Lancer l'indexation de tous les fichiers audio
    print("📌 Indexation en cours...")
    djv.fingerprint_directory(AUDIO_DIR, [".mp3", ".wav"])
    print("✅ Tous les fichiers audio ont été indexés avec succès.")
