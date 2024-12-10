from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv  # Import load_dotenv
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset
import numpy as np
import tensorflow as tf

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



# Load OpenOrca dataset
dataset = load_dataset("Open-Orca/OpenOrca") # FILE NOT FOUND ERROR
dataset = dataset['train'].select(range(10000))

#  Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def preprocess_function(examples):
    inputs = ["question: " + example for example in examples["question"]]
    targets = [example if len(example) > 0 else "I don't know" for example in examples["response"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding=True, return_tensors="np")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True, padding=True, return_tensors="np")
    model_inputs["labels"] = labels["input_ids"].astype(np.int32)
    model_inputs["decoder_input_ids"] = labels["input_ids"].astype(np.int32)
    model_inputs["input_ids"] = labels["input_ids"].astype(np.int32)
    model_inputs["attention_mask"] = labels["input_ids"].astype(np.int32)
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format("tf")

# prepare the tensorflow dataset

model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

tf_dataset = tokenized_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels", "decoder_input_ids"],
    shuffle=True,
    batch_size=128,
    # drop_remainder=True  ensures all batches have the same size
)

# fine tune it

for layer in model.layers[:-1]:
    layer.trainable = False

learning_rate = 1e-2  # Replace `le-2` with `1e-2`

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.summary()


# def adjust_prompt_based_on_sentiment(prompt):
#     sentiment = sentiment_analysis(prompt)



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
    context = "\n".join(conversation_history[-5:])  # Only keep the last 5 exchanges

    try:
        # Tokenize the input context
        inputs = tokenizer(context, return_tensors="tf", max_length=512, truncation=True, padding=True)

        # Generate a response using the model
        outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Append bot response to conversation history
        conversation_history.append(f"Assistant: {response}")

        return jsonify({'message': response})
    except Exception as e:
        print(f"Model error: {e}")
        return jsonify({'response': "Sorry, I couldn't process your request."}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear the conversation history."""
    global conversation_history
    conversation_history = []
    return jsonify({'message': 'Conversation history cleared.'})


if __name__ == '__main__':
    app.run(port=5000)