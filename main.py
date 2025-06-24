from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transcribe import transcrire_et_stocker   # ← doit matcher exactement

app = FastAPI()

class TranscriptionRequest(BaseModel):
    doc_id: str
    path_audio: str

@app.post("/transcribe")
async def transcribe(req: TranscriptionRequest):
    try:
        transcrire_et_stocker(req.path_audio, req.doc_id)
        return {"status": "success", "doc_id": req.doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))