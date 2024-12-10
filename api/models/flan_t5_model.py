from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Load the Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
