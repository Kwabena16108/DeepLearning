import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import datetime as datetime
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, ParameterGrid
import copy
# suppress the PerformanceWarning
warnings.simplefilter('ignore')

#---------------------------------------------------------------------------------#

# Define the LSTM model
class StackedLSTMModel(nn.Module):
        
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, hidden_layer_dim=100, dropout=0.5):
        super(StackedLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_layer_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x):
        batch_size, seq_length, num_features, num_subfeatures = x.size()
        x = x.view(batch_size, seq_length * num_features, num_subfeatures)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out


class BidirectionalLSTMModel(nn.Module):
        
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, hidden_layer_dim=100, dropout=0.5):
        super(BidirectionalLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_layer_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x):
        batch_size, seq_length, num_features, num_subfeatures = x.size()
        x = x.view(batch_size, seq_length * num_features, num_subfeatures)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out


def load_dataset():

    training_features = pd.read_hdf('aapl_spy_training_examples.h5', key='data')
    test_features = pd.read_hdf('aapl_spy_test_examples.h5', key='data')

    training_target = pd.read_hdf('aapl_spy_training_target.h5', key='data')
    test_target = pd.read_hdf('aapl_spy_test_target.h5', key='data')

    training_target = torch.eye(3)[training_target.values]
    test_target = torch.eye(3)[test_target.values]
    
    # Get a list of the technical indicators
    technical_keys = np.unique([i[0] for i in training_features.keys()])

    # Build a dictionary from the loaded training_feature DataFrame
    training_technical_dict = {}
    for technical in technical_keys:
        training_technical_dict[technical] = training_features[technical]

    # Build a dictionary from the loaded test_feature DataFrame
    test_technical_dict = {}
    for technical in technical_keys:
        test_technical_dict[technical] = test_features[technical]

    training_shape = (16, 6764, 2)
    test_shape = (16, 756, 2)

    training_features_tensor = torch.zeros(training_shape)
    test_features_tensor = torch.zeros(test_shape)

    for index, key in enumerate(technical_keys): 
        training_features_tensor[index, :, :].copy_(torch.tensor(training_technical_dict[key].values)) 
    
    for index, key in enumerate(technical_keys):
        test_features_tensor[index, :, :].copy_(torch.tensor(test_technical_dict[key].values)) 
    
    return training_features_tensor, test_features_tensor, training_target, test_target

class StockDataset(Dataset):
    def __init__(self, stock_tensor, stock_target_labels_df, example_length):
        self.stock_tensor = stock_tensor
        self.example_length = example_length
        self.stock_targets = stock_target_labels_df

    def __len__(self):
        # The number of examples in the stock dataset
        return self.stock_tensor.shape[1] - self.example_length + 1

    def __getitem__(self, this_index):
        start_index = this_index
        end_index = this_index + self.example_length
        
        # I'm doing the transpose here so that the dimensionality goes: 
        # number of features : # of stocks : # of dates
        features = self.stock_tensor[:,start_index:end_index,:].clone().detach().transpose(1,2)
        label = torch.squeeze(self.stock_targets[this_index].clone().detach(), -1)

        return features, label

def create_sequences(input_data, target_data, seq_length):
    xs = []
    ys = []

    for i in range(len(input_data) - seq_length):
        x = input_data[i:i + seq_length]
        y = target_data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# Train the model
def train(model, criterion, optimizer, train_data, train_targets, seq_length, num_epochs, device):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch, (inputs, targets) in enumerate(train_data):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            # print(inputs.size())
            # print(targets.size())
            outputs = model(inputs)
            # # print(outputs)
            # # print(targets)
            _, predicted_labels = torch.max(outputs,1)
            # # targets = targets.view(-1)
            one_hot_output = torch.zeros(outputs.shape).to(outputs.device)
            one_hot_output.scatter_(1, predicted_labels.unsqueeze(1), 1)
            # print(one_hot_output)
            # print(outputs.view(-1))
            # print(predicted_labels)
            # loss = criterion(one_hot_output.view(-1), targets.view(-1))
            # one_hot_targets = to_one_hot(targets, 3)
            loss = criterion(outputs, targets.squeeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss #/ len(train_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def test(model, criterion, test_data, test_targets, seq_length, device):
    model.eval()
    model.to(device)
    
    running_loss = 0.0
    predicted_all = []
    actual_all = []

    with torch.no_grad():
        for inputs, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(1))
            running_loss += loss.item()

            predicted_all.extend(torch.argmax(outputs, dim=1).tolist())
            actual_all.extend(targets.tolist())

    test_loss = running_loss / len(test_data)
    return np.array(predicted_all), np.array(actual_all), test_loss


class BootstrappedEnsembleModel(nn.Module):
    def __init__(self, models):
        super(BootstrappedEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        out = torch.mean(torch.stack(outputs), dim=0)
        return out

def train_bagging(ensemble_model, criterion, optimizers, train_data, train_targets, seq_length, num_epochs, device):
    for model, optimizer in zip(ensemble_model.models, optimizers):
        model = train(model, criterion, optimizer, train_data, seq_length, num_epochs, device)
    return ensemble_model.models


def tune_hyperparameters(train_data, test_data, train_targets, test_targets, seq_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    param_grid = {
        "model_class": [StackedLSTMModel, BidirectionalLSTMModel],
        "model_params": [
            {"input_dim": 14, "hidden_dim": 50, "num_layers": 3, "output_dim": 3},
            {"input_dim": 14, "hidden_dim": 100, "num_layers": 3, "output_dim": 3},
        ],
        "num_folds": [3],
        "loss_fn": [nn.BCEWithLogitsLoss()],
        "learning_rate": [0.0001],
        "epochs": [5],
    }

    best_estimator = None
    best_params = None
    best_error = float("inf")
    cv_results = []

    for params in ParameterGrid(param_grid):
        model = params["model_class"](**params["model_params"])

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        validation_error = rolling_window_cv(model, train_data, test_data, train_targets, test_targets, seq_length, params["loss_fn"], optimizer,device, params["num_folds"], params["epochs"])
        
        cv_results.append({"params": params, "validation_error": validation_error})

        if validation_error < best_error:
            best_error = validation_error
            best_estimator = model
            best_params = params

    return best_estimator, best_params, cv_results

def rolling_window_cv(model, train_data, test_data, train_targets, test_targets, seq_length, loss_fn, optimizer, device, num_folds, epochs):
    test_data_size = len(test_data.dataset)
    n = test_data_size // num_folds
    validation_errors = []

    for fold in range(num_folds):
        # Deep copy the optimizer to have a fresh instance for each fold
        optimizer_copy = copy.deepcopy(optimizer)
        model_copy = copy.deepcopy(model)

        print(f"Fold {fold+1}/{num_folds}")

        start_idx = fold * n
        end_idx = (fold + 1) * n

        test_data_fold = torch.utils.data.Subset(test_data.dataset, range(start_idx, end_idx))

        train_data_fold_indices = list(range(0, start_idx)) + list(range(end_idx, test_data_size))
        train_data_fold = torch.utils.data.Subset(train_data.dataset, train_data_fold_indices)

        test_loader = DataLoader(test_data_fold, batch_size=test_data.batch_size, shuffle=False)
        train_loader = DataLoader(train_data_fold, batch_size=train_data.batch_size, shuffle=True)

        
        criterion = loss_fn

        train(model_copy, criterion, optimizer_copy, train_data, train_targets, seq_length, epochs, device)

        predicted, actual, r_loss = test(model_copy, criterion, test_loader, test_targets, seq_length, device)
        
        print(f"Fold {fold+1} Test Loss: {r_loss}")
        # print(predicted)
        error = np.mean(predicted != actual)
        validation_errors.append(error)

    return np.mean(validation_errors)


def main():
    # Load and preprocess the dataset
    batch_size = 128
    training_features_tensor, test_features_tensor, train_targets, test_targets = load_dataset()
    train_data = StockDataset(training_features_tensor, train_targets, 14)
    test_data = StockDataset(test_features_tensor, test_targets, 14)
    stock_training_dataloader =DataLoader(train_data, batch_size=batch_size, shuffle=True)
    stock_test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Set parameters
    seq_length = 10  # Change this value based on your dataset

    # Tune hyperparameters
    best_estimator, best_params, cv_results = tune_hyperparameters(stock_training_dataloader, stock_test_dataloader, train_targets, test_targets, seq_length)

    # Create the stacked LSTM and bidirectional LSTM models using the best hyperparameters
    if best_params["model_params"]["hidden_dim"]:
        hidden_dim = best_params["model_params"]["hidden_dim"]
    else:
        hidden_dim = 64

    stacked_lstm_model = StackedLSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=3, output_dim=3)
    bidirectional_lstm_model = BidirectionalLSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=3, output_dim=3)

    # Train the models with bootstrapping
    models = [stacked_lstm_model, bidirectional_lstm_model]
    criterion = nn.MSELoss()
    optimizers = [torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"]) for model in models]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # trained_models = train_bagging(models, criterion, optimizers, train_data, train_targets, seq_length, best_params["epochs"], device)

    # Create the bootstrapped ensemble model
    ensemble_model = BootstrappedEnsembleModel(models)
    trained_models = train_bagging(models, criterion, optimizers, train_data, train_targets, seq_length, best_params["epochs"], device)
    ensemble_model.models = nn.ModuleList(trained_models)
    # Test the model
    predicted, actual = test(ensemble_model, test_data, seq_length)

    # De-normalize the predictions and actual values
    min_val, max_val = np.min(train_data), np.max(train_data)
    predicted_denorm = predicted * (max_val - min_val) + min_val
    actual_denorm = actual * (max_val - min_val) + min_val

    # Plot the results
    plt.plot(predicted_denorm, label="Predicted")
    plt.plot(actual_denorm, label="Actual")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()