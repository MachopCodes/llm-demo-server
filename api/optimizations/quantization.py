import tensorflow as tf

def quantize_model(model, save_path="flan_t5_quantized.tflite"):
    model.save("saved_model/flan_t5_pruned")
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/flan_t5_pruned")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for _ in range(100):
            yield [tf.random.uniform(shape=[1, 512], dtype=tf.int32)]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(save_path, "wb") as f:
        f.write(tflite_model)

    print(f"Quantized model saved at: {save_path}")


def quantize_model(model, model_name="model", save_dir="saved_model", save_path=None):
    """
    Quantizes a given TensorFlow model and saves it as a TFLite file.

    Args:
        model: The TensorFlow model to be quantized.
        model_name (str): Name of the model (used for directory naming).
        save_dir (str): Directory where the model is saved in SavedModel format.
        save_path (str): Path to save the quantized TFLite model. Defaults to '<model_name>_quantized.tflite'.

    Returns:
        None
    """
    if not save_path:
        save_path = f"{model_name}_quantized.tflite"

    # Save the model in SavedModel format
    model_save_dir = f"{save_dir}/{model_name}"
    model.save(model_save_dir)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Define a representative dataset for quantization
    def representative_dataset_gen():
        for _ in range(100):
            yield [tf.random.uniform(shape=[1, 512], dtype=tf.int32)]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert and save the quantized model
    tflite_model = converter.convert()
    with open(save_path, "wb") as f:
        f.write(tflite_model)

    print(f"Quantized model saved at: {save_path}")
