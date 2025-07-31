"""
Task-specific classification heads for GLUE benchmark evaluation.

This module implements proper classification heads for each GLUE task type,
replacing the simple heuristics used in the initial evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np


class GLUEClassificationHead(nn.Module):
    """Base class for GLUE task classification heads."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Handle sequence inputs by taking mean
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Apply dropout and classification
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        return logits


class RTEClassificationHead(GLUEClassificationHead):
    """Classification head for Recognizing Textual Entailment (RTE)."""
    
    def __init__(self, input_dim: int):
        super().__init__(input_dim, num_classes=2)  # entailment, not_entailment
        
    def predict(self, hidden_states: torch.Tensor) -> Tuple[str, float]:
        """
        Predict entailment with confidence.
        
        Returns:
            prediction: "entailment" or "not_entailment"
            confidence: prediction confidence [0, 1]
        """
        logits = self.forward(hidden_states)
        probs = F.softmax(logits, dim=-1)
        
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = torch.max(probs, dim=-1)[0].item()
        
        prediction = "entailment" if predicted_class == 1 else "not_entailment"
        return prediction, confidence


class WNLIClassificationHead(GLUEClassificationHead):
    """Classification head for Winograd Natural Language Inference (WNLI)."""
    
    def __init__(self, input_dim: int):
        super().__init__(input_dim, num_classes=2)  # entailment, not_entailment
        
    def predict(self, hidden_states: torch.Tensor) -> Tuple[str, float]:
        """
        Predict Winograd inference with confidence.
        
        Returns:
            prediction: "entailment" or "not_entailment"
            confidence: prediction confidence [0, 1]
        """
        logits = self.forward(hidden_states)
        probs = F.softmax(logits, dim=-1)
        
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = torch.max(probs, dim=-1)[0].item()
        
        prediction = "entailment" if predicted_class == 1 else "not_entailment"
        return prediction, confidence


class SentimentClassificationHead(GLUEClassificationHead):
    """Classification head for Sentiment Analysis (SST-2 style)."""
    
    def __init__(self, input_dim: int):
        super().__init__(input_dim, num_classes=2)  # positive, negative
        
    def predict(self, hidden_states: torch.Tensor) -> Tuple[str, float]:
        """
        Predict sentiment with confidence.
        
        Returns:
            prediction: "positive" or "negative"
            confidence: prediction confidence [0, 1]
        """
        logits = self.forward(hidden_states)
        probs = F.softmax(logits, dim=-1)
        
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = torch.max(probs, dim=-1)[0].item()
        
        prediction = "positive" if predicted_class == 1 else "negative"
        return prediction, confidence


class GLUEClassificationManager:
    """Manager for GLUE task classification heads."""
    
    def __init__(self, input_dim: int, device: str = "cpu"):
        self.input_dim = input_dim
        self.device = device
        
        # Initialize classification heads
        self.heads = {
            "rte": RTEClassificationHead(input_dim).to(device),
            "wnli": WNLIClassificationHead(input_dim).to(device),
            "sentiment": SentimentClassificationHead(input_dim).to(device)
        }
        
        # Training state
        self.training_mode = False
        
    def get_head(self, task_name: str) -> GLUEClassificationHead:
        """Get classification head for specific task."""
        if task_name not in self.heads:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.heads.keys())}")
        return self.heads[task_name]
    
    def predict(self, task_name: str, hidden_states: torch.Tensor) -> Tuple[str, float]:
        """Make prediction for specific task."""
        head = self.get_head(task_name)
        return head.predict(hidden_states)
    
    def train_head(self, 
                   task_name: str, 
                   examples: list, 
                   learning_rate: float = 1e-3,
                   epochs: int = 10) -> Dict[str, Any]:
        """
        Train classification head on examples.
        
        Args:
            task_name: Name of the task
            examples: List of (hidden_states, label) pairs
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            
        Returns:
            training_info: Training statistics
        """
        head = self.get_head(task_name)
        head.train()
        
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for hidden_states, label in examples:
                # Convert label to tensor
                if isinstance(label, str):
                    if task_name in ["rte", "wnli"]:
                        label_idx = 1 if label == "entailment" else 0
                    elif task_name == "sentiment":
                        label_idx = 1 if label == "positive" else 0
                    else:
                        raise ValueError(f"Unknown label format for task {task_name}")
                else:
                    label_idx = label
                
                label_tensor = torch.tensor([label_idx], dtype=torch.long, device=self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = head(hidden_states)
                loss = criterion(logits, label_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                predicted = torch.argmax(logits, dim=-1)
                correct += (predicted == label_tensor).sum().item()
                total += 1
            
            avg_loss = epoch_loss / len(examples)
            accuracy = correct / total
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        head.eval()
        
        return {
            "final_loss": losses[-1],
            "final_accuracy": accuracies[-1],
            "losses": losses,
            "accuracies": accuracies,
            "epochs": epochs
        }
    
    def evaluate_head(self, task_name: str, test_examples: list) -> Dict[str, float]:
        """Evaluate classification head on test examples."""
        head = self.get_head(task_name)
        head.eval()
        
        correct = 0
        total = 0
        confidences = []
        
        with torch.no_grad():
            for hidden_states, true_label in test_examples:
                prediction, confidence = head.predict(hidden_states)
                
                if prediction == true_label:
                    correct += 1
                total += 1
                confidences.append(confidence)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_confidence": avg_confidence
        }
