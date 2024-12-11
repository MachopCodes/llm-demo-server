import torch.nn.functional as F
from torch.nn import KLDivLoss

# Define the distillation loss function
# This combines knowledge distillation loss (KL divergence) with the standard cross-entropy loss.
def distillation_loss(student_logits, teacher_logits, labels, temperature=1.0):
    """
    Computes the distillation loss for training a student model with a teacher model.

    Args:
        student_logits: Logits output by the student model (un-normalized predictions).
        teacher_logits: Logits output by the teacher model (un-normalized predictions).
        labels: Ground truth labels for the data.
        temperature: Temperature parameter for softening the logits during distillation.

    Returns:
        Total loss: A combination of KL divergence loss and cross-entropy loss.
    """
    # Convert teacher logits to probabilities using softmax and the given temperature.
    # Higher temperature smooths the teacher's probability distribution.
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Convert student logits to log probabilities using log_softmax and the given temperature.
    # Log probabilities are used for calculating KL divergence.
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    # Compute the KL divergence loss between the softened distributions of teacher and student.
    # KLDivLoss measures how the student diverges from the teacher's predictions.
    # The loss is scaled by the square of the temperature.
    kl_loss = KLDivLoss(reduction="batchmean")(student_probs, teacher_probs) * (temperature**2)
    
    # Compute the cross-entropy loss between the student's predictions and the ground truth labels.
    # Labels are flattened for comparison with the student logits, and -100 is ignored.
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)), 
        labels.view(-1), 
        ignore_index=-100
    )
    
    # Return the combined loss (distillation + cross-entropy).
    return kl_loss + ce_loss
