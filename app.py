# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv  # Import load_dotenv

load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv('OPENAI_API_KEY')  # Set your API key

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing


# Initialize a conversation history list
conversation_history = []

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('input', '')
    
    # Append user input to conversation history
    conversation_history.append({'role': 'user', 'content': user_input})
    
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=conversation_history
        )
        message = response['choices'][0]['message']['content']
        
        # Append assistant's response to conversation history
        conversation_history.append({'role': 'assistant', 'content': message})
        
        return jsonify({'message': message})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while communicating with OpenAI API.'}), 500

if __name__ == '__main__':
    app.run(port=5000)