import torch
from main_analyze import train_model, train_accuracies
if __name__ == "__main__":
    print("Pretraining the model...")
    train_model()  # saves the model to "models/plant_disease_model.pth"
    print(f"Training complete. Final accuracy: {train_accuracies[-1]}")