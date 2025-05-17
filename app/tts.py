# from elevenlabs import generate, save, set_api_key
from elevenlabs.client import ElevenLabs
from .config import ELEVENLABS_API_KEY
from elevenlabs import VoiceSettings
import os
from io import BytesIO
from elevenlabs import play



def text_to_speech(text: str) -> bytes:
    print('entered text to speech')
    client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    print('client created')
    response = client.text_to_speech.convert(
        text=text,
        voice_id="9BWtsMINqrJLrRacOk9x",
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )
    # Create a BytesIO object to hold the audio data in memory
    audio_stream = BytesIO()
    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    # Reset stream position to the beginning
    audio_stream.seek(0)

    return audio_stream