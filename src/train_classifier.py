import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from src.data_preparation import prepare_data
from src.generate_synthetic_data import generate_synthetic_data

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def train_classifier(generator, num_synthetic_samples=1000, num_epochs=50, batch_size=64, learning_rate=0.001):
    X_train, X_test, y_train, y_test = prepare_data()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(generator, num_synthetic_samples)

    # Combine synthetic data with real data
    combined_X_train = torch.cat((X_train_tensor, synthetic_data), 0)
    combined_y_train = torch.cat((y_train_tensor, torch.ones(num_synthetic_samples, 1)), 0)

    # Define the classifier
    input_dim = X_train.shape[1]
    classifier = SimpleClassifier(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # Training loop for the classifier
    for epoch in range(num_epochs):
        for i in range(0, combined_X_train.size(0), batch_size):
            inputs = combined_X_train[i:i+batch_size]
            labels = combined_y_train[i:i+batch_size]

            outputs = classifier(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return classifier, X_test_tensor, y_test_tensor