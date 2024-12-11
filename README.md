# Lightweight LLM Chatbot

This repository is a Python-based demonstration of a lightweight and deployable chatbot. The chatbot leverages optimization techniques such as distillation, pruning, and quantization to create an efficient and performant language model suitable for deployment in resource-constrained environments.

## Features

- **Distillation:** A student model is trained to mimic the behavior of a larger teacher model, reducing model size while retaining performance.
- **Pruning:** Redundant weights in the model are removed to decrease model size and computation.
- **Quantization:** Model weights are converted to lower precision (e.g., INT8) for improved inference speed and reduced memory usage.
- **Chatbot Functionality:** Lightweight chatbot capable of generating conversational responses.

## Project Structure

```
api/
├── data/                   # Raw and preprocessed datasets
│   ├── raw/                # Raw datasets (e.g., downloaded datasets)
│   ├── processed/          # Preprocessed/tokenized datasets
│
├── models/                 # Models and related utilities
│   ├── flan_t5_model.py    # Model loading and tokenizer setup
│   ├── student_model/      # Optimized student model
│   ├── teacher_model/      # Teacher model for distillation
│
├── optimizations/          # Optimization scripts
│   ├── pruning.py          # Pruning utilities
│   ├── quantization.py     # Quantization utilities
│   ├── distillation.py     # Distillation training loop
│
├── scripts/                # Utility scripts for running tasks
│   ├── run_distillation.py # Script for running distillation
│   ├── evaluate_model.py   # Script for evaluating model performance
│   ├── prepare_model.py    # Script for preparing models for deployment
│
├── services/               # Chatbot service
│   ├── chatbot_service.py  # Core chatbot functionality
│
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
└── .gitignore              # Git ignore rules
```

## Requirements

- Python 3.8+
- TensorFlow
- PyTorch
- Hugging Face Transformers
- TensorFlow Model Optimization Toolkit
- Datasets

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Run Distillation
Distillation trains a student model to mimic the teacher model.

```bash
python api/scripts/run_distillation.py
```

The trained student model will be saved to `api/models/student_model/`.

### 2. Evaluate the Model
Evaluate the model's performance using the ROUGE metric on a subset of the validation dataset:

```bash
python api/scripts/evaluate_distillation.py
```

### 3. Apply Pruning
Apply pruning to the model to reduce its size:

```bash
python api/scripts/prepare_model.py
```

### 4. Quantize the Model
Quantize the pruned model for deployment:

```bash
python api/scripts/prepare_model.py
```

The quantized model will be saved as a `.tflite` file for lightweight deployment.

### 5. Use the Chatbot
Run the chatbot service to interact with the optimized language model:

```python
from api.services.chatbot_service import generate_response

# Example usage
conversation_history = []
response = generate_response("Hello, how are you?", conversation_history)
print(response)
```

## Optimization Details

### Distillation
- Teacher Model: `google/flan-t5-base`
- Student Model: `google/flan-t5-small`
- Trained using a subset of the CNN/DailyMail dataset.

### Pruning
Prunes dense and convolutional layers using TensorFlow's Polynomial Decay schedule.

### Quantization
Quantizes the model to INT8 precision using TensorFlow Lite for fast inference.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.