import tensorflow as tf
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # Replace with your local tokenizer path if needed

# Load the TensorFlow SavedModel
model = tf.saved_model.load("saved_model/flan_t5_pruned")

# Retrieve the serving function
serving_function = model.signatures["serving_default"]

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
    context = "\n".join(conversation_history[-5:])

    # Step 3: Tokenize the prepared context
    inputs = tokenizer(
        context,
        return_tensors="tf",
        max_length=512,
        truncation=True,
        padding=True
    )

    # Step 4: Prepare the inputs for the model
    input_data = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": inputs["input_ids"],  # Assuming same input for decoder in this setup
        "decoder_attention_mask": inputs["attention_mask"],  # Assuming same attention mask for decoder
    }

    # Step 5: Run the model and decode the output logits
    outputs = serving_function(**input_data)
    logits = outputs["logits"]  # Access the logits from the output
    predicted_ids = tf.argmax(logits, axis=-1)  # Select the token with the highest probability

    # Step 6: Decode the predicted tokens to text
    response = tokenizer.decode(predicted_ids.numpy()[0], skip_special_tokens=True)

    # Step 7: Add the model's response to the conversation history
    conversation_history.append(f"Assistant: {response}")

    # Step 8: Return the generated response
    return response
