import sys
import os
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from api.optimizations.pruning import apply_pruning_to_layers  # Import the pruning utility
from api.optimizations.quantization import quantize_model  # Import the quantization utility

def prepare_model_for_deployment():
    """
    Prepares the student model for deployment by applying pruning and quantization.

    Steps:
        1. Load the student model saved after distillation.
        2. Apply pruning to reduce the size of the model while maintaining accuracy.
        3. Compile the pruned model for further optimization.
        4. Quantize the model to create an efficient, deployment-ready version.

    Returns:
        None
    """
    # Path to the saved student model from the distillation step
    student_model_path = "api/models/student_model"

    # Step 1: Load the student model using Hugging Face Transformers
    print("Loading the student model...")
    if not os.path.exists(student_model_path):
        raise FileNotFoundError(f"Student model not found at {student_model_path}. Did you run distillation?")

    model = TFAutoModelForSeq2SeqLM.from_pretrained(student_model_path)
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)

    # Step 2: Apply pruning to the student model
    print("Applying pruning...")
    pruned_model = apply_pruning_to_layers(model)

    # Step 3: Compile the pruned model
    print("Compiling the pruned model...")
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Use Adam optimizer with a learning rate of 1e-4
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Loss for classification tasks
    )

    # Step 4: Quantize the pruned model
    print("Quantizing the pruned model...")
    quantized_model_path = "api/models/student_model_quantized.tflite"
    quantize_model(pruned_model, save_path=quantized_model_path)

    print(f"Model is ready for deployment! Quantized model saved at {quantized_model_path}")

if __name__ == "__main__":
    # Entry point for the script
    prepare_model_for_deployment()
