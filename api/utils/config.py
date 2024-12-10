import os

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ALLOWED_ORIGINS = [
        "http://localhost:3000", 
        "http://localhost:3001", 
        "https://machopcodes.github.io",
        "http://127.0.0.1:5000/"
    ]
