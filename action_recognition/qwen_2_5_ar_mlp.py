import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# 1. Paths and Configurations
# ---------------------------
train_dir = ""
val_dir   = ""
output_model_path = ""
confusion_matrix_path = ""

NUM_CLASSES = 100         # 100 action classes
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_DIM = 512          # Hidden layer size of the MLP head
EMBEDDING_DIM = 3584      # Provided embedding dimension

# For the Transformer, reshape each embedding (3584,) into a sequence of length 16 (T=16) with token dimension 224 (D=224)
SEQUENCE_LENGTH = 16
TOKEN_DIM = 224           # 16 * 224 = 3584

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2. Custom Dataset Class
# ---------------------------
class VideoEmbeddingDataset(Dataset):
    def __init__(self, root_dir):
        """
        Expects each subfolder in root_dir to represent an action class.
        Each subfolder should contain .npy files holding the embedding vector.
        Each embedding is a 1D vector of shape (3584,), reshaped to (SEQUENCE_LENGTH, TOKEN_DIM).
        """
        self.files = []
        self.labels = []
        # List all subdirectories and sort them (this will be our class order)
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.num_features = EMBEDDING_DIM

        for class_name in self.classes:
            label_path = os.path.join(root_dir, class_name)
            # Use the index in the sorted list as the label (0 to NUM_CLASSES-1)
            numeric_label = self.classes.index(class_name)
            for file in os.listdir(label_path):
                if file.endswith(".npy"):
                    file_path = os.path.join(label_path, file)
                    self.files.append(file_path)
                    self.labels.append(numeric_label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        # Load the embedding from the .npy file
        data = np.load(file_path)
        flat_embedding = data.flatten()
        assert flat_embedding.shape[0] == SEQUENCE_LENGTH * TOKEN_DIM, f"Unexpected embedding shape: {flat_embedding.shape}"
        feature_sequence = flat_embedding.reshape(SEQUENCE_LENGTH, TOKEN_DIM)
        return torch.tensor(feature_sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ---------------------------
# 3. Load Datasets and DataLoaders
# ---------------------------
train_dataset = VideoEmbeddingDataset(train_dir)
val_dataset = VideoEmbeddingDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Loaded dataset | Each sample: ({SEQUENCE_LENGTH}, {TOKEN_DIM}) | Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
print("Classes:", train_dataset.classes)

# ---------------------------
# 4. MLP Classifier Model
# ---------------------------
# This version uses a simple MLP (without transformer layers) for action recognition.
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# For the MLP, we simply average-pool the sequence tokens to get a single vector
class ActionMLP(nn.Module):
    def __init__(self, token_dim, seq_len, hidden_dim, num_classes):
        super(ActionMLP, self).__init__()
        self.fc = MLPClassifier(token_dim, hidden_dim, num_classes)
    def forward(self, x):
        # x: (batch, seq_len, token_dim)
        x = x.mean(dim=1)  # Average pooling over the sequence dimension -> (batch, token_dim)
        return self.fc(x)

model = ActionMLP(TOKEN_DIM, SEQUENCE_LENGTH, HIDDEN_DIM, NUM_CLASSES).to(device)

# Optionally, if class frequencies are imbalanced, you can compute and pass class weights.
# For now, we'll use the default loss.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------
# 5. Training and Evaluation Functions
# ---------------------------
def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_path)
    print(f"Confusion Matrix saved at {confusion_matrix_path}")

def evaluate():
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            # For the MLP, features is of shape (batch, seq_len, token_dim)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, digits=4)
    print("Classification Report:\n", report)
    plot_confusion_matrix(all_labels, all_preds)
    return val_acc

def train():
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_acc = evaluate()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_model_path)
            print(f"Model saved! Best Val Acc: {best_val_acc:.4f}")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")

# ---------------------------
# 6. Main Execution
# ---------------------------
if __name__ == "__main__":
    train()
