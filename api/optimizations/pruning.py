import tensorflow_model_optimization as tfmot
import tensorflow as tf

def apply_pruning_to_layers(model):
    """
    Applies pruning to compatible layers of a TensorFlow model to reduce its size and improve efficiency.

    Args:
        model: A TensorFlow model to which pruning will be applied.

    Returns:
        model: The TensorFlow model with pruning applied to compatible layers.
    """
    # Define the pruning schedule: sparsity gradually increases during training
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,  # Start with no sparsity
        final_sparsity=0.5,    # End with 50% sparsity
        begin_step=0,          # Start pruning at step 0
        end_step=1000          # End pruning by step 1000
    )

    # Iterate through each layer in the model
    for i, layer in enumerate(model.layers):
        # Check if the layer is compatible with pruning (Dense or Conv2D layers)
        if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Pruning layer {i}: {layer.name}")
            # Replace the original layer with a pruned version
            model.layers[i] = tfmot.sparsity.keras.prune_low_magnitude(
                layer, pruning_schedule=pruning_schedule
            )
        else:
            # Skip layers that are not compatible with pruning
            print(f"Skipping layer {i}: {layer.name} (not compatible)")

    return model
