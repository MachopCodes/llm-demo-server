from api.models.flan_t5_model import model, tokenizer

def generate_response(user_input, conversation_history):
    # Add user input to conversation history
    conversation_history.append(f"User: {user_input}")

    # Prepare the conversation history
    context = "\n".join(conversation_history[-5:])

    # Tokenize the input context
    inputs = tokenizer(context, return_tensors="tf", max_length=512, truncation=True, padding=True)

    # Generate a response using the model
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Append bot response to conversation history
    conversation_history.append(f"Assistant: {response}")

    return response
