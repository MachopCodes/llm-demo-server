from api.models.flan_t5_model import model, tokenizer

def generate_response(user_input, conversation_history):
    """
    Generates a response to a user's input based on the conversation history.

    Args:
        user_input (str): The latest user input to respond to.
        conversation_history (list): List of past conversation exchanges.

    Returns:
        str: The model-generated response.
    """
    # Step 1: Add the user's input to the conversation history
    conversation_history.append(f"User: {user_input}")

    # Step 2: Prepare the context from the last 5 exchanges in the conversation history
    # This limits the input context length for efficiency and relevance
    context = "\n".join(conversation_history[-5:])

    # Step 3: Tokenize the prepared context
    # Tokenization includes truncation and padding to fit the model's input requirements
    inputs = tokenizer(
        context,
        return_tensors="tf",  # Generate TensorFlow-compatible tensors
        max_length=512,       # Limit input length to 512 tokens
        truncation=True,      # Truncate input if it exceeds the maximum length
        padding=True          # Pad input to ensure consistent length
    )

    # Step 4: Use the model to generate a response
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50)  # Generate up to 50 new tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode generated tokens to a readable string

    # Step 5: Add the model's response to the conversation history
    conversation_history.append(f"Assistant: {response}")

    # Step 6: Return the generated response
    return response
