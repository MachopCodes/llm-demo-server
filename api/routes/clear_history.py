from flask import Blueprint, jsonify

clear_history_bp = Blueprint('clear_history', __name__)

@clear_history_bp.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear the conversation history."""
    global conversation_history
    conversation_history.clear()
    return jsonify({'message': 'Conversation history cleared.'})
