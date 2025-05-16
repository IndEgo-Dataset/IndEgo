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
# 4. Transformer Encoder Classifier with CLS Token
# ---------------------------
class TransformerClassifierWithCLS(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, num_classes, num_heads, num_layers, dropout):
        """
        input_dim: token dimension (e.g., 224)
        seq_len: number of tokens (e.g., 16)
        """
        super(TransformerClassifierWithCLS, self).__init__()
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        # Learnable positional encoding for the sequence (CLS + tokens)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len + 1, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(input_dim)
        # MLP head for classification
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        # Expand CLS token for each sample in the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: (batch_size, 1, input_dim)
        # Prepend CLS token to the input sequence
        x = torch.cat((cls_tokens, x), dim=1)  # shape: (batch_size, seq_len+1, input_dim)
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)
        # Normalize the output
        x = self.layer_norm(x)
        # Use the CLS token (first token) for classification
        cls_output = x[:, 0, :]  # shape: (batch_size, input_dim)
        return self.fc(cls_output)

model = TransformerClassifierWithCLS(TOKEN_DIM, SEQUENCE_LENGTH, 512, 100, 2, 2, 0.1).to(device)

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
            loss = nn.CrossEntropyLoss()(outputs, labels)
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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train()
