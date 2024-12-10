from flask import Blueprint, request, jsonify
from api.services.chatbot_service import generate_response

chatbot_bp = Blueprint('chatbot', __name__)

conversation_history = []

@chatbot_bp.route('/chatbot', methods=['POST'])
def chatbot():
    global conversation_history

    # Get user input
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': 'Please provide a message.'}), 400

    # Generate a response using the service
    response = generate_response(user_input, conversation_history)

    return jsonify({'message': response})