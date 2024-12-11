import sys
import os

# Add the root directory of the project to sys.path
# This allows the script to import modules from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from api.models.flan_t5_model import model  # Import the T5 model instance
from api.optimizations.pruning import apply_pruning_to_layers  # Import the pruning utility
from api.optimizations.quantization import quantize_model  # Import the quantization utility
import tensorflow as tf

def prepare_model_for_deployment():
    """
    Prepares the model for deployment by applying pruning and quantization.

    Steps:
        1. Apply pruning to reduce the size of the model while maintaining accuracy.
        2. Compile the pruned model for further optimization.
        3. Quantize the model to create an efficient, deployment-ready version.

    Returns:
        None
    """
    # Step 1: Apply pruning to the model
    print("Applying pruning...")
    pruned_model = apply_pruning_to_layers(model)

    # Step 2: Compile the pruned model with an optimizer and loss function
    print("Compiling pruned model...")
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Use Adam optimizer with a learning rate of 1e-4
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Loss for classification tasks
    )

    # Step 3: Apply quantization to convert the model to a more efficient format
    print("Quantizing the pruned model...")
    quantize_model(pruned_model)

    print("Model is ready for deployment!")

if __name__ == "__main__":
    # Entry point for the script
    # Executes the model preparation process
    prepare_model_for_deployment()
