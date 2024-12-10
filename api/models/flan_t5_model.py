from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load the Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Pruning Configuration
def apply_pruning_to_layers(model):
    import tensorflow_model_optimization as tfmot

    # Define pruning parameters
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=1000
    )

    # Iterate through the model's layers
    for i, layer in enumerate(model.layers):
        # Prune only layers that are compatible
        if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Pruning layer {i}: {layer.name}")
            model.layers[i] = tfmot.sparsity.keras.prune_low_magnitude(
                layer, pruning_schedule=pruning_schedule
            )
        else:
            print(f"Skipping layer {i}: {layer.name} (not compatible)")

    return model

# Apply pruning to the model's layers
model = apply_pruning_to_layers(model)

# Compile the pruned model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)


#  include pruning in models/flan_t5_model.py 
#  because that's where the model is defined and initialized. 
#  By doing this, the model is pruned when it's first loaded.