from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv  # Import load_dotenv
from transformers import pipeline

load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv('OPENAI_API_KEY')  # Set your API key

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000", 
    "http://localhost:3001", 
    "https://machopcodes.github.io",
    "http://127.0.0.1:5000/"
    ], supports_credentials=True)



# Initialize a conversation history list
conversation_history = []

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('input', '')
    
    # Append user input to conversation history
    conversation_history.append({'role': 'user', 'content': user_input})
    
    try:
        result = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=conversation_history)
        message = result['choices'][0]['message']['content']
        
        # Append assistant's response to conversation history
        conversation_history.append({'role': 'assistant', 'content': message})
        
        return jsonify({'message': message})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while communicating with OpenAI API.'}), 500
    
    

# Load the Hugging Face model

# Model	Size	Optimized for Conversation	Performance
# DialoGPT-small	~117M parameters	Yes	Good for lightweight
# distilgpt2	~82M parameters	No	Decent with prompts
# BlenderBot-400M-distill	~400M parameters	Yes	Better conversations
# T5-small / flan-t5-small	~60M parameters	No	Flexible but basic



# Load DialoGPT-small model for text-generation pipeline
chatbot_model = pipeline("text-generation", model="microsoft/DialoGPT-small")

# Global conversation history dictionary (per user, if needed in the future)
conversation_history = []

@app.route('/chatbot', methods=['POST'])
def chatbot():
    global conversation_history

    # Get user input
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': 'Please provide a message.'}), 400

    # Add user input to conversation history
    conversation_history.append(f"User: {user_input}")

    # Prepare the conversation history as a single string
    context = "\n".join(conversation_history[-5:])  # Limit to last 5 exchanges for efficiency

    try:
        # Generate a response using the model
        result = chatbot_model(context, max_new_tokens=50)
        response = result[0]['generated_text']

        # Append bot response to conversation history
        conversation_history.append(f"Assistant: {response}")

        return jsonify({'message': response})
    except Exception as e:
        print(f"Model error: {e}")
        return jsonify({'response': "Sorry, I couldn't process your request."})


@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear the conversation history."""
    global conversation_history
    conversation_history = []
    return jsonify({'message': 'Conversation history cleared.'})


if __name__ == '__main__':
    app.run(port=5000)