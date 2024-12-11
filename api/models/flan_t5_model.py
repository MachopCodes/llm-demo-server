from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from api.optimizations.pruning import apply_pruning_to_layers

# Load the Hugging Face tokenizer for the FLAN-T5-small model.
# The tokenizer is responsible for converting text to input IDs and vice versa.
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Load the pre-trained FLAN-T5-small model from Hugging Face.
# This is a T5 model specialized for sequence-to-sequence tasks.
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Apply pruning to the model's layers using a custom function.
# Pruning reduces the number of weights in the model, potentially improving inference speed and reducing size.
model = apply_pruning_to_layers(model)

# Compile the pruned model with an optimizer and loss function.
# - Optimizer: Adam optimizer with a learning rate of 1e-4, used to update model weights during training.
# - Loss: Sparse Categorical Crossentropy, designed for classification tasks with integer labels.
#   'from_logits=True' indicates that the model outputs raw logits.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
