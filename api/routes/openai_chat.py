from flask import Blueprint, request, jsonify
import openai

openai_chat_bp = Blueprint('openai_chat', __name__)

@openai_chat_bp.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('input', '')

    try:
        result = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': user_input}]
        )
        message = result['choices'][0]['message']['content']
        return jsonify({'message': message})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while communicating with OpenAI API.'}), 500
