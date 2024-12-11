import tensorflow as tf

def quantize_model(model, save_path="flan_t5_quantized.tflite"):
    """
    Quantizes a TensorFlow model using TensorFlow Lite and saves it as a `.tflite` file.

    Args:
        model: The TensorFlow model to be quantized.
        save_path (str): The file path to save the quantized model. Defaults to 'flan_t5_quantized.tflite'.

    Returns:
        None
    """
    # Save the model in SavedModel format for conversion
    model.save("saved_model/flan_t5_pruned")

    # Initialize a TFLite converter from the SavedModel directory
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/flan_t5_pruned")

    # Enable default optimizations for TensorFlow Lite conversion
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Define a representative dataset generator for post-training quantization
    def representative_dataset_gen():
        for _ in range(100):
            # Generate random input data with the same shape as the model expects
            yield [tf.random.uniform(shape=[1, 512], dtype=tf.int32)]

    # Assign the representative dataset generator to the converter
    converter.representative_dataset = representative_dataset_gen

    # Specify that the model should use INT8 quantization for both input and output
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert the model to TFLite format
    tflite_model = converter.convert()

    # Save the quantized model to the specified file path
    with open(save_path, "wb") as f:
        f.write(tflite_model)

    print(f"Quantized model saved at: {save_path}")


def quantize_model_generic(model, model_name="model", save_dir="saved_model", save_path=None):
    """
    A generic function to quantize a TensorFlow model and save it as a `.tflite` file.

    Args:
        model: The TensorFlow model to be quantized.
        model_name (str): Name of the model (used for naming the SavedModel directory).
        save_dir (str): Directory where the model is saved in SavedModel format.
        save_path (str): Path to save the quantized TFLite model. If not provided, defaults to '<model_name>_quantized.tflite'.

    Returns:
        None
    """
    # Set the default save path if none is provided
    if not save_path:
        save_path = f"{model_name}_quantized.tflite"

    # Construct the directory path for saving the model
    model_save_dir = f"{save_dir}/{model_name}"

    # Save the model in SavedModel format
    model.save(model_save_dir)

    # Initialize a TFLite converter from the SavedModel directory
    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_dir)

    # Enable default optimizations for TensorFlow Lite conversion
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Define a representative dataset generator for post-training quantization
    def representative_dataset_gen():
        for _ in range(100):
            # Generate random input data with the same shape as the model expects
            yield [tf.random.uniform(shape=[1, 512], dtype=tf.int32)]

    # Assign the representative dataset generator to the converter
    converter.representative_dataset = representative_dataset_gen

    # Specify that the model should use INT8 quantization for both input and output
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert the model to TFLite format
    tflite_model = converter.convert()

    # Save the quantized model to the specified file path
    with open(save_path, "wb") as f:
        f.write(tflite_model)

    print(f"Quantized model saved at: {save_path}")
