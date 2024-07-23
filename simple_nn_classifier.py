import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(csvFileName):
    dataframe = pd.read_csv(csvFileName)
    dataframe = dataframe.drop(labels=['filename', 'index'], axis=1)
    labels = dataframe['isMalicious']
    data = dataframe.loc[:, dataframe.columns != 'isMalicious']
    
    return data, labels

def split_dataset(data, labels, test_ratio):
    training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size=test_ratio, random_state=42)
    for f in training_data.columns:
        encoder = LabelEncoder().fit(data[f])
        training_data[f] = encoder.transform(training_data[f]).astype(np.float32)
        test_data[f] = encoder.transform(test_data[f]).astype(np.float32)
    # Convert data to PyTorch tensors
    training_data = torch.tensor(training_data.to_numpy(), dtype=torch.float32)
    test_data = torch.tensor(test_data.to_numpy(), dtype=torch.float32)
    training_labels = torch.tensor(training_labels.to_numpy(), dtype=torch.float32).view(-1, 1)
    test_labels = torch.tensor(test_labels.to_numpy(), dtype=torch.float32).view(-1, 1)

    return training_data, test_data, training_labels, test_labels

# Create a simple feed-forward neural network for tabular data
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc0 = nn.Linear(input_dim, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Training the model
def train_model(training_data, test_data, training_labels, test_labels, modelpath, epochs = 500):
    input_dim = training_data.shape[1]
    model = SimpleNN(input_dim)
    best_model = None
    best_accuracy = 0.0
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):  # Loop over the dataset multiple times
        optimizer.zero_grad()
        outputs = model(training_data)
        loss = criterion(outputs, training_labels)
        loss.backward()
        optimizer.step()
        # Step the scheduler
        #scheduler.step()

        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                pred_labels_prob = model(test_data)
                pred_labels = (pred_labels_prob > 0.5).float()
    
            accuracy = accuracy_score(test_labels, pred_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                state = {
                    'state_dict': model.state_dict(),
                    'input_dim': input_dim
                }
                torch.save(state, modelpath)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f} Accuracy: {accuracy:.4f}')
    
    print(f'----Finished Training: (Accuracy:{best_accuracy:.4f})----')
    print(f'Trained model is saved to file: {modelpath}\n')

# Evaluate the model
def evaluate_model(filename, test_data, test_labels):
    state = torch.load(filename)
    model = SimpleNN(state['input_dim'])
    model.load_state_dict(state['state_dict'])
    model.eval()
    print(f'\nModel loaded from {filename}')
    with torch.no_grad():
        pred_labels_prob = model(test_data)
        pred_labels = (pred_labels_prob > 0.5).float()
    
    accuracy = accuracy_score(test_labels, pred_labels)
    roc_auc = roc_auc_score(test_labels, pred_labels_prob)
    conf_matrix = confusion_matrix(test_labels, pred_labels)
    class_report = classification_report(test_labels, pred_labels)
    
    print(f'Accuracy: {accuracy:.4f}\n')
    print(f'ROC AUC: {roc_auc:.4f}\n')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('\nClassification Report:')
    print(class_report)

# Main execution
if __name__ == '__main__':
    model_path = "simple-nn-classifier.model"
    data, labels = preprocess_data('./dataset/features.csv')
    training_data, test_data, training_labels, test_labels = split_dataset(data, labels, 0.25)
    train_model(training_data, test_data, training_labels, test_labels, model_path)
    evaluate_model(model_path, test_data, test_labels)

