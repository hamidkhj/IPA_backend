import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
TOGATHERAI_API_KEY = os.getenv('TOGATHERAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
