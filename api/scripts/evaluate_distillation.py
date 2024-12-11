import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import evaluate

def evaluate_model(model_path, dataset_name, split="validation[:1%]"):
    """
    Evaluates a trained T5 model on a specified dataset and computes ROUGE scores.

    Args:
        model_path (str): Path to the directory containing the trained model.
        dataset_name (str): Name of the dataset to evaluate on.
        split (str): The split of the dataset to use (e.g., 'validation[:1%]'). Defaults to 1% of the validation set.

    Returns:
        dict: A dictionary containing the computed ROUGE scores.
    """
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model and its tokenizer from the specified path
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Load the specified dataset and evaluation metric (ROUGE in this case)
    dataset = load_dataset(dataset_name, "3.0.0", split=split)
    rouge = evaluate.load("rouge")

    # Iterate over the dataset to evaluate the model
    for batch in dataset:
        # Tokenize the input text for the model
        inputs = tokenizer(
            batch["article"],  # Input article text
            truncation=True,  # Truncate input to avoid exceeding max length
            padding="max_length",  # Pad inputs to the max length
            max_length=512,  # Set the maximum input length
            return_tensors="pt"  # Return PyTorch tensors
        ).to(device)

        # Generate predictions for the input data
        with torch.no_grad():  # Disable gradient computation for evaluation
            outputs = model.generate(
                input_ids=inputs["input_ids"],  # Input token IDs
                attention_mask=inputs["attention_mask"],  # Attention mask
                max_length=128  # Maximum length of the generated output
            )

        # Decode the generated token IDs into text
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract the reference summaries
        references = [batch["highlights"]]

        # Add the predictions and references to the ROUGE metric for computation
        rouge.add_batch(predictions=predictions, references=references)

    # Compute the ROUGE scores across all evaluated examples
    results = rouge.compute()

    # Print and return the evaluation results
    print("Evaluation Results:", results)
    return results

if __name__ == "__main__":
    # Run the evaluation using the student model on the CNN/DailyMail dataset
    evaluate_model("api/models/student_model", "cnn_dailymail")


# 12-11-2024 Evaluation Results: 
# 'rouge1': 0.12041369528099011,   
#       Measures the overlap of unigrams (single words) between 
#       the predicted and reference texts
# 'rouge2': 0.04015715085214712,   
#       Measures the overlap of bigrams (two consecutive words).
# 'rougeL': 0.09846018269836967,   
#       Focuses on the longest common subsequence (LCS) between 
#       predicted and reference texts, accounting for sequence ordering.
# 'rougeLsum': 0.1095096397586956} 
#       A variation of ROUGE-L used specifically for summarization tasks.

# The evaluation results indicate how well your student model performed based on the ROUGE metric. 
# Here’s a breakdown: 0.1095: About 10.95% similarity in summary-like results.