from fastapi import APIRouter
from pydantic import BaseModel
from .rag import answer_question
from .rag import generate_answer_from_transcription
from fastapi import UploadFile, File
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

router = APIRouter()

class Question(BaseModel):
    query: str

@router.post("/ask")
async def ask_question(question: Question):
    answer = await answer_question(question.query)
    return {"answer": answer}





# UPLOAD_DIR = "uploaded_audio"
# os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create dir if not exists

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
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


    # use transcription to generate answer
    answer = await generate_answer_from_transcription(transcript)


    # audio_path = text_to_speech(answer)
    # audio_url = f"/audio/{os.path.basename(audio_path)}"

    return {"transcription": transcript, "answer": answer}



@router.post("/speak")
async def speak_text(data: dict):
    text = data.get("text", "")
    audio = text_to_speech(text)
    print(type(audio))
    return StreamingResponse(audio, media_type="audio/mpeg")