from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

def create_app():
    load_dotenv()  # Load environment variables

    app = Flask(__name__)
    CORS(app, origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "https://machopcodes.github.io",
        "http://127.0.0.1:5000/"
    ], supports_credentials=True)

    # Register blueprints
    from .routes.chatbot import chatbot_bp
    from .routes.openai_chat import openai_chat_bp
    from .routes.clear_history import clear_history_bp

    app.register_blueprint(chatbot_bp)
    app.register_blueprint(openai_chat_bp)
    app.register_blueprint(clear_history_bp)

    return app