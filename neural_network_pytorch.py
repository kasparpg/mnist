"""
PyTorch Neural Network for MNIST Digit Classification
Optimized for high accuracy using modern deep learning techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time


class MNISTNet(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.
    Architecture optimized for high accuracy on digit recognition.
    """
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 2 pooling operations: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Third conv block (no pooling)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout_conv(x)
        
        # Flatten
        x = x.view(-1, 128 * 7 * 7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        x = self.fc3(x)
        return x


class MNISTTrainer:
    """Handles training and evaluation of the MNIST neural network."""
    
    def __init__(self, batch_size=128, learning_rate=0.001, epochs=20):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Data transforms with augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Initialize model
        self.model = MNISTNet().to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=469,  # ~60000/128
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        self.train_loader = None
        self.test_loader = None
        
    def load_data(self):
        """Load MNIST dataset using torchvision."""
        print("\nLoading MNIST dataset...")
        
        # Get project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, 'data', 'pytorch_mnist')
        
        # Download and load training data
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Download and load test data
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type in ['cuda', 'mps'] else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type in ['cuda', 'mps'] else False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print("Data loaded successfully.\n")
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"\r  Batch {batch_idx + 1}/{len(self.train_loader)} - "
                      f"Loss: {running_loss / (batch_idx + 1):.4f} - "
                      f"Acc: {100. * correct / total:.2f}%", end='')
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        print(f"\r  Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%")
        return train_loss, train_acc
    
    def evaluate(self):
        """Evaluate model on test set."""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        print(f"  Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%")
        return test_loss, test_acc
    
    def train(self):
        """Full training loop."""
        print("=" * 50)
        print("MNIST PyTorch Neural Network Training")
        print("=" * 50)
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 50)
        
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            print("-" * 30)
            
            epoch_start = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            test_loss, test_acc = self.evaluate()
            epoch_time = time.time() - epoch_start
            
            print(f"  Epoch time: {epoch_time:.1f}s")
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_model('best_model.pth')
                print(f"  -> New best model saved! ({test_acc:.2f}%)")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 50)
        print("Training Complete!")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best test accuracy: {best_acc:.2f}%")
        print("=" * 50)
        
        return best_acc
    
    def save_model(self, filename):
        """Save model weights."""
        project_root = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(project_root, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        filepath = os.path.join(weights_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        
    def load_model(self, filename):
        """Load model weights."""
        project_root = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(project_root, 'weights', filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {filename}")
            return True
        return False
    
    def predict(self, image):
        """Predict digit for a single image."""
        self.model.eval()
        with torch.no_grad():
            if len(image.shape) == 2:
                image = image.unsqueeze(0).unsqueeze(0)
            elif len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()
        return prediction, confidence


def print_info():
    """Print system and model information."""
    print("\n" + "=" * 50)
    print("MNIST PyTorch Neural Network")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 50)


if __name__ == '__main__':
    print_info()
    
    # Create trainer with optimized hyperparameters
    trainer = MNISTTrainer(
        batch_size=128,
        learning_rate=0.001,
        epochs=20
    )
    
    # Load data
    trainer.load_data()
    
    # Train the model
    best_accuracy = trainer.train()
    
    # Final evaluation
    print("\nFinal Evaluation:")
    trainer.evaluate()
