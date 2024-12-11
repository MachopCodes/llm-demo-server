import torch

from .distillation_loss import distillation_loss

def train_distillation(teacher_model, student_model, data_loader, optimizer, device, epochs=3, temperature=1.0):
    """
    Trains the student model using knowledge distillation from the teacher model.

    Args:
        teacher_model: Pre-trained teacher model.
        student_model: Student model to be trained.
        data_loader: DataLoader providing the training batches.
        optimizer: Optimizer for updating student model parameters.
        device: Device to use for computation (e.g., "cuda" or "cpu").
        epochs: Number of training epochs.
        temperature: Temperature for softening logits in distillation.

    Returns:
        None
    """
    # Set teacher model to evaluation mode and move it to the specified device
    teacher_model.eval().to(device)
    # Set student model to training mode and move it to the specified device
    student_model.train().to(device)

    for epoch in range(epochs):
        for batch in data_loader:
            # Move input data and labels to the specified device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Prepare decoder inputs and labels for sequence generation
            decoder_input_ids = labels[:, :-1].contiguous()  # Remove last token
            labels = labels[:, 1:].contiguous()  # Shift labels to ignore padding at start

            # Compute teacher model's predictions without gradient computation
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )

            # Compute student model's predictions
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )

            # Compute the distillation loss
            loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, labels, temperature)

            # Zero the gradients, backpropagate, and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the loss for the current batch
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")