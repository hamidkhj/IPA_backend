from fastapi import APIRouter
from pydantic import BaseModel
from .rag import answer_question
from .rag import generate_answer_from_transcription
from .rag import store_user_document
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
import httpx
import os
from datetime import datetime
from fastapi.responses import Response
from .tts import text_to_speech
import io
from io import BytesIO
from .config import DEEPGRAM_API_KEY
from PyPDF2 import PdfReader
from docx import Document
import json


router = APIRouter()

class Question(BaseModel):
    query: str

@router.post("/ask")
async def ask_question(question: Question):
    answer = await answer_question(question.query)
    return {"answer": answer}



@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".txt"):
            text = content.decode("utf-8")
        elif filename.endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        elif filename.endswith(".docx"):
            from docx import Document
            doc = Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

        success = store_user_document(text)
        if success:
            return {"message": "Document indexed"}
        return JSONResponse(content={"error": "Failed to process document"}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# UPLOAD_DIR = "uploaded_audio"
# os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create dir if not exists

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), history: str = Form(default=[])):
    # # Save file locally (Optional)
    # timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # filename = f"{timestamp}_{file.filename}"
    # file_path = os.path.join(UPLOAD_DIR, filename)

    # with open(file_path, "wb") as f:
    #     content = await file.read()
    #     f.write(content)


    # send without writing the file
    content = await file.read()

    # Send to Deepgram
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": file.content_type  # Typically audio/mp3 or audio/webm
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.deepgram.com/v1/listen",
            content=content,
            headers=headers
        )

    transcript = response.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")

    conversation_history = json.loads(history)
    
    # use transcription to generate answer
    answer = await generate_answer_from_transcription(transcript, conversation_history)

    # audio_path = text_to_speech(answer)
    # audio_url = f"/audio/{os.path.basename(audio_path)}"

    return {"transcription": transcript, "answer": answer}



@router.post("/speak")
async def speak_text(data: dict):
    text = data.get("text", "")
    audio = text_to_speech(text)
    print(type(audio))
    return StreamingResponse(audio, media_type="audio/mpeg")



@router.get("/ping")
def ping():
    return {"status": "ok"}