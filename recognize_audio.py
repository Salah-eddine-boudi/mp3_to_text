# recognize_audio.py
from dejavu import Dejavu  
from dejavu.recognize import FileRecognizer  




import json
import sys
import os

# Charger la configuration MySQL
with open("dejavu.cnf") as f:
    config = json.load(f)

# Initialiser Dejavu
djv = Dejavu(config)

# Vérifier l'argument du fichier extrait
if len(sys.argv) < 2:
    print("❌ Utilisation : python recognize_audio.py chemin/vers/extrait.wav")
    sys.exit(1)

extrait_path = sys.argv[1]

# Vérifier que le fichier existe
if not os.path.exists(extrait_path):
    print("❌ Fichier audio introuvable.")
    sys.exit(1)

# Lancer la reconnaissance
print(f"🎧 Reconnaissance de : {extrait_path}")
song = djv.recognize(FileRecognizer, extrait_path)

# Afficher le résultat
if song is None:
    print("🔍 Aucun fichier correspondant trouvé.")
else:
    print(f"✅ Fichier reconnu : {song['song_name']} (confiance : {song['confidence']})")
