import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for digits 0 to 9

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output for the fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Set device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        #print(f"item loss per batch: { loss.item() }")
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training information
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

# Evaluation on the test set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    test_loss = 0
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        loss = criterion(outputs, targets)  # Use the same criterion as during training
        test_loss += loss.item()

    accuracy = correct / total
    print(f"Accuracy on the test set: {accuracy:.4f}")
    print(f'Test Accuracy: {100 * accuracy:.2f}%')
    print(f'Test Loss: {test_loss}')

with torch.no_grad():
    correct = 0
    total = 0
    train_loss = 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        loss = criterion(outputs, targets)  # Use the same criterion as during training
        train_loss += loss.item()

    accuracy = correct / total
    print(f"Accuracy on the train set: {accuracy:.4f}")
    print(f'Train Accuracy: {100 * accuracy:.2f}%')
    print(f'Train Loss: {train_loss}')

# Save the trained model
checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': accuracy,
}

torch.save(checkpoint, 'models/trained_model.pth')
print("Trained model saved successfully.")
'''
# Load the saved model
loaded_model = SimpleCNN().to(device)
loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.001)

checkpoint = torch.load('trained_model.pth')
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loaded_epoch = checkpoint['epoch']
loaded_accuracy = checkpoint['accuracy']

# Set the loaded model to evaluation mode (if needed)
loaded_model.eval()
'''