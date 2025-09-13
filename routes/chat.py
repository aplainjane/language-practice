from flask import Blueprint, request, jsonify
from chat_logic import call_deepseek

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    response = call_deepseek(user_input)
    return jsonify({"reply": response})