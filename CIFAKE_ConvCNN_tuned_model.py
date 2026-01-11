import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class CIFAKE_ConvCNN_Tuned(nn.Module):
    def __init__(self, n_neurons=64, dropout_p=0.5):
        super(CIFAKE_ConvCNN_Tuned, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 8 * 8, n_neurons)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(n_neurons, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

def get_data_loaders(root, batch_size=64):
    training_data = datasets.ImageFolder(root=os.path.join(root, 'train'),
                                         transform=ToTensor())
    test_data = datasets.ImageFolder(root=os.path.join(root, 'test'),
                                         transform=ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def train_and_evaluate(model, train_loader, val_loader, epochs, lr, weight_decay, device):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        epoch_train_acc = correct_train / total_train
        epoch_train_loss = running_loss / len(train_loader)

        model.eval()
        correct_val, total_val, running_val_loss = 0, 0, 0.0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)

            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_loop.set_postfix(loss=loss.item())

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

    return history, best_val_acc

def tuner(root_dir, device, n_samples_for_check=2):
    params_for_testing = {
        'batch_size': list(range(32, 129, 32)),
        'n_neurons': [64, 128, 256],
        'dropout_min': 0.2,
        'dropout_max': 0.5,
        'lr': [0.001, 0.0005, 0.0001],
        'weight_decay': [0.0, 1e-3, 1e-4, 1e-5]
    }

    best_config, best_global_acc = None, 0.0
    results_log = []

    start_time = time.time()

    for i in range(n_samples_for_check):
        config = {
            'batch_size': random.choice(params_for_testing['batch_size']),
            'n_neurons': random.choice(params_for_testing['n_neurons']),
            'dropout_p': round(random.uniform(params_for_testing['dropout_min'], params_for_testing['dropout_max']), 2),
            'lr': random.choice(params_for_testing['lr']),
            'weight_decay': random.choice(params_for_testing['weight_decay'])
        }

        print(f"\n[{i + 1}/{n_samples_for_check}] Testing configuration: {config}")

        model = CIFAKE_ConvCNN_Tuned(
            n_neurons=config['n_neurons'],
            dropout_p=config['dropout_p']
        ).to(device)

        train_loader, test_loader = get_data_loaders(root_dir, config['batch_size'])
        history, final_val_acc = train_and_evaluate(
            model, train_loader, test_loader,
            epochs=3,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            device=device
        )

        print(f"/t Validation accuracy: {final_val_acc:.4f}")

        results_log.append((config, final_val_acc))
        if final_val_acc > best_global_acc:
            best_global_acc = final_val_acc
            best_config = config

    print(f"\n\nBEST Configuration:\n {best_config} \n\n")

    final_model = CIFAKE_ConvCNN_Tuned(
        n_neurons=best_config['n_neurons'],
        dropout_p=best_config['dropout_p']
    ).to(device)

    train_loader, test_loader = get_data_loaders(root_dir, best_config['batch_size'])
    history, final_val_acc = train_and_evaluate(
        final_model, train_loader, test_loader,
        epochs=10,
        lr=best_config['lr'],
        weight_decay=best_config['weight_decay'],
        device=device
    )

    print(f"\n\nBEST Model Accuracy: {final_val_acc:.4f}")
    print(f"Time taken for training: {time.time() - start_time:.2f}s \n\n")

    return final_model, test_loader, history

def get_metrics_and_plot(model, test_loader, history, device="mps"):
    os.makedirs('training_results/conv_cnn_tuner', exist_ok=True)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Precision:\t{precision:.4f}")
    print(f"Recall:\t{recall:.4f}")
    print(f"F1 Score:\t{f1:.4f}")

    plt.figure(figsize=(30, 10))

    #Loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    #Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Validation Acc')
    plt.title('Model Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    #Confusion Matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig('training_results/conv_cnn_tuner/conv_cnn_tuner.png')
    plt.show()

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    best_model, test_loader, history = tuner('dataset', DEVICE, n_samples_for_check=10)

    get_metrics_and_plot(best_model, test_loader, history, DEVICE)

    torch.save(best_model.state_dict(), 'training_results/conv_cnn_tuner/conv_cnn_tuner.pth')