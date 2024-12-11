import sys
import os

# Add the root directory to sys.path to allow importing from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from api.optimizations.distillation import train_distillation

def main():
    """
    Main function to perform model distillation.

    Steps:
        1. Load teacher and student models.
        2. Preprocess the dataset for tokenization.
        3. Train the student model using knowledge distillation.
        4. Save the trained student model and tokenizer.
    """
    # Configuration parameters
    config = {
        "teacher_model": "google/flan-t5-base",  # Pretrained teacher model
        "student_model": "google/flan-t5-small",  # Pretrained student model
        "dataset": "cnn_dailymail",  # Dataset to use for training
        "batch_size": 8,  # Number of samples per batch
        "epochs": 3,  # Number of training epochs
        "learning_rate": 5e-5,  # Learning rate for optimizer
        "temperature": 2.0  # Temperature for distillation
    }

    # Determine the device to use (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load teacher and student models and tokenizer
    print("Loading models and tokenizer...")
    teacher_model = T5ForConditionalGeneration.from_pretrained(config["teacher_model"]).to(device)
    student_model = T5ForConditionalGeneration.from_pretrained(config["student_model"]).to(device)
    tokenizer = T5Tokenizer.from_pretrained(config["teacher_model"])

    # Step 2: Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    dataset = load_dataset(config["dataset"], "3.0.0", split="train[:1%]")

    def preprocess(batch):
        """
        Tokenizes and formats the dataset for training.

        Args:
            batch (dict): A batch of data from the dataset.

        Returns:
            dict: Tokenized input IDs, attention masks, and labels.
        """
        inputs = tokenizer(batch["article"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        labels = tokenizer(batch["highlights"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }

    tokenized_dataset = dataset.map(preprocess, batched=True)
    tokenized_dataset.set_format(type="torch")  # Format dataset for PyTorch

    # Step 3: Create DataLoader for batching
    print("Creating DataLoader...")
    data_loader = DataLoader(tokenized_dataset, batch_size=config["batch_size"], shuffle=True)

    # Step 4: Set up optimizer
    print("Setting up optimizer...")
    optimizer = AdamW(student_model.parameters(), lr=config["learning_rate"])

    # Step 5: Train the student model using distillation
    print("Starting distillation training...")
    train_distillation(
        teacher_model,
        student_model,
        data_loader,
        optimizer,
        device,
        epochs=config["epochs"],
        temperature=config["temperature"]
    )


    # Step 6: Save the trained student model and tokenizer
    print("Saving the trained student model and tokenizer...")
    save_path = "./api/models/student_model"  # Define the save path
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists
    student_model.save_pretrained(save_path) # Save the model and tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"Distillation complete! Student model and tokenizer saved to '{save_path}'")

if __name__ == "__main__":
    # Entry point for the script
    main()
