from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Wrapper for traditional scikit-learn models
class SklearnModelWrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model
        self.feature_importances_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        if hasattr(self.model, "coef_"):  # For linear models
            self.feature_importances_ = self.model.coef_.flatten()
        elif hasattr(self.model, "feature_importances_"):  # For tree-based models
            self.feature_importances_ = self.model.feature_importances_
        else:
            self.feature_importances_ = None

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            # Handle cases where probabilities are 1D (e.g., binary classification)
            if probas.ndim == 1:
                return probas
            return probas[:, 1]  # Return probabilities for the positive class
        else:
            raise NotImplementedError("This model does not support probability predictions.")

# Wrapper for PyTorch neural networks
class PytorchModelWrapper(BaseEstimator):
    def __init__(self, input_dim, layers, learning_rate=0.001, batch_size=32, epochs=50, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(input_dim, layers)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs

    def _build_model(self, input_dim, layers):
        model = nn.Sequential()
        for i, neurons in enumerate(layers):
            if i == 0:
                model.add_module(f"Layer_{i+1}", nn.Linear(input_dim, neurons))
            else:
                model.add_module(f"Layer_{i+1}", nn.Linear(layers[i-1], neurons))
            model.add_module(f"ReLU_{i+1}", nn.ReLU())
            model.add_module(f"Dropout_{i+1}", nn.Dropout(0.2))
        model.add_module("Output", nn.Linear(layers[-1], 1))
        model.add_module("Sigmoid", nn.Sigmoid())
        return model.to(self.device)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch).view(-1)  # Ensure shape [batch_size]
                # y_pred = self.model(X_batch).squeeze()
                # loss = self.criterion(y_pred, y_batch)
                loss = self.criterion(y_pred, y_batch.view(-1))  # Match target shape
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).cpu().numpy().flatten()
        return (y_pred > 0.5).astype(int)

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).cpu().numpy().flatten()
        return y_pred
