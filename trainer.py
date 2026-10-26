"""Shared PyTorch training loop used by AttenPCA-Net and the shallow MLP."""

from datetime import datetime
from time import time_ns
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Classifier:
    def train(self, x_train, y_train):
        pass

    def test(self, x_test):
        pass


class MetricsCalculator:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.predictions = []
        self.true_labels = []

    def update(self, predictions, labels):
        self.predictions.extend(predictions.cpu().numpy())
        self.true_labels.extend(labels.cpu().numpy())

    def compute_confusion_matrix(self):
        return confusion_matrix(self.true_labels, self.predictions)

    def compute_f1_macro(self):
        return f1_score(self.true_labels, self.predictions, average="macro")


class TorchClassifier(nn.Module, Classifier):
    """Train a module with Adam, cross-entropy, and early stopping on loss.

    Twenty percent of the training segments are held out for validation when
    a validation split is not passed in. The checkpoint with the lowest
    validation loss is restored at the end of training.
    """

    def get_model(self, **kwargs) -> nn.Module: ...

    def get_optimizer(self): ...

    def get_scheduler(self, optimizer):
        return None

    def get_loss(self): ...

    def model_train_call(self, data):
        return self.model(data)

    def model_test_call(self, data):
        return self.model(data)

    def set_patience(self):
        return 30

    def calculate_model_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
        return total_params

    def set_batch_size(self):
        return 32

    def __init__(self, **kwargs):
        super(TorchClassifier, self).__init__()
        self.identifier = time_ns()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.get_model(**kwargs)
        self.calculate_model_parameters()
        self.criterion = self.get_loss()
        print("DEVICE")
        print(self.device)
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer)
        self.batch_size = self.set_batch_size()
        self.patience = self.set_patience()

    def train(self, x_train, y_train, x_val=None, y_val=None):
        if x_val is None or y_val is None:
            train_data, valid_data, train_targets, valid_targets = train_test_split(
                x_train, y_train, test_size=0.2, stratify=y_train
            )
        else:
            train_data, valid_data, train_targets, valid_targets = (
                x_train,
                x_val,
                y_train,
                y_val,
            )

        train_data_tensor = torch.tensor(train_data.astype(np.float32))
        train_targets_tensor = torch.tensor(train_targets.astype(np.int64))
        train_dataset = TensorDataset(train_data_tensor, train_targets_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        valid_data_tensor = torch.tensor(valid_data.astype(np.float32))
        valid_targets_tensor = torch.tensor(valid_targets.astype(np.int64))
        valid_dataset = TensorDataset(valid_data_tensor, valid_targets_tensor)
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=True
        )

        early_stopper = EarlyStopper(patience=self.patience)
        epoch = 0
        while True:
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.valid_one_epoch(valid_loader)
            if self.scheduler:
                self.scheduler.step(val_metrics["f1_macro"])

            print(f"\nEpoch [{epoch + 1}]:")
            print("Training Metrics:")
            print(f'Loss: {train_metrics["loss"]:.4f}')
            print(f'Accuracy: {train_metrics["accuracy"]:.2f}%')
            print(f'F1 Macro: {train_metrics["f1_macro"]:.4f}')
            print("Confusion Matrix:")
            print(train_metrics["confusion_matrix"])
            print("_______________________________")
            print("Validation Metrics:")
            print(f'Loss: {val_metrics["loss"]:.4f}')
            print(f'Accuracy: {val_metrics["accuracy"]:.2f}%')
            print(f'F1 Macro: {val_metrics["f1_macro"]:.4f}')
            print("Confusion Matrix:")
            print(val_metrics["confusion_matrix"])
            print("====================================================")
            print("====================================================")
            epoch += 1
            if early_stopper.early_stop(val_metrics["loss"], self):
                break
        self.save_final_model(early_stopper.max_model)
        self.load_final()

    def test(self, x_test):
        test_data_tensor = torch.tensor(x_test.astype(np.float32))
        test_dataset = TensorDataset(test_data_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.model.to(self.device)
        self.model.eval()

        result = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(self.device)
                outputs = self.model_test_call(inputs)
                result.append(outputs.cpu().numpy())

        return np.vstack(result)

    def train_one_epoch(self, train_dataloader):
        metrics = MetricsCalculator()
        self.model = self.model.to(self.device)
        self.model.train()

        running_loss = 0.0
        total_samples = 0

        for batch_idx, (data, target) in tqdm(
            enumerate(train_dataloader),
            desc="Training",
            total=len(train_dataloader),
            leave=True,
            ncols=80,
        ):
            batch_size = data.size(0)
            total_samples += batch_size
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model_train_call(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * batch_size

            _, predicted = torch.max(output.data, 1)
            metrics.update(predicted, target)

        avg_loss = running_loss / total_samples
        confusion_mat = metrics.compute_confusion_matrix()
        f1_macro = metrics.compute_f1_macro()
        accuracy = 100 * np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "confusion_matrix": confusion_mat,
            "metrics_calculator": metrics,
        }

    def valid_one_epoch(self, valid_dataloader):
        metrics = MetricsCalculator()
        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to(self.device)

            running_loss = 0.0
            total_samples = 0

            for batch_idx, (data, target) in tqdm(
                enumerate(valid_dataloader),
                desc="Validating",
                total=len(valid_dataloader),
                leave=True,
                ncols=80,
            ):
                batch_size = data.size(0)
                total_samples += batch_size
                data, target = data.to(self.device), target.to(self.device)
                output = self.model_test_call(data)
                loss = self.criterion(output, target)
                running_loss += loss.item() * batch_size

                _, predicted = torch.max(output.data, 1)
                metrics.update(predicted, target)

        avg_loss = running_loss / total_samples
        confusion_mat = metrics.compute_confusion_matrix()
        f1_macro = metrics.compute_f1_macro()
        accuracy = 100 * np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "confusion_matrix": confusion_mat,
            "metrics_calculator": metrics,
        }

    def _checkpoint_name(self, model_name, suffix):
        return (
            f"{model_name}_checkpoints/{model_name}_{self.identifier}_{suffix}.pt"
        )

    def save_checkpoint(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (
            self.model.model_name
            if hasattr(self.model, "model_name")
            else self.model.__class__.__name__
        )
        filename = self._checkpoint_name(model_name, f"epoch_{timestamp}")
        checkpoint = {
            "model_name": model_name,
            "model_state_dict": self.model.state_dict(),
            "timestamp": timestamp,
        }
        os.makedirs(f"{model_name}_checkpoints", exist_ok=True)
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
        return filename

    def save_final_model(self, file_name=None):
        if file_name:
            self.model = self.load_checkpoint(file_name)

        model_name = self.model.__class__.__name__
        filename = self._checkpoint_name(model_name, "final_version")
        checkpoint = {
            "model_name": model_name,
            "model_state_dict": self.model.state_dict(),
        }
        os.makedirs(f"{model_name}_checkpoints", exist_ok=True)
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
        return filename

    def load_checkpoint(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No checkpoint found at {filename}")

        checkpoint = torch.load(filename, map_location="cpu")
        current_model_name = self.model.__class__.__name__
        saved_model_name = checkpoint.get("model_name", "")
        if current_model_name != saved_model_name:
            print(
                f"Warning: Loading checkpoint from different model class "
                f"(Saved: {saved_model_name}, Current: {current_model_name})"
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return self.model

    def load_final(self):
        model_name = self.model.__class__.__name__
        filename = self._checkpoint_name(model_name, "final_version")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No checkpoint found at {filename}")

        checkpoint = torch.load(filename, map_location="cpu")
        current_model_name = self.model.__class__.__name__
        saved_model_name = checkpoint.get("model_name", "")
        if current_model_name != saved_model_name:
            print(
                f"Warning: Loading checkpoint from different model class "
                f"(Saved: {saved_model_name}, Current: {current_model_name})"
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return self.model


class EarlyStopper:
    def __init__(self, patience=30):
        self.patience = patience
        self.counter = 0
        self.max_val_f1_score_macro = 0
        self.max_model = None
        self.best_loss = float("inf")

    def early_stop(self, comparison_metric, model: TorchClassifier):
        if comparison_metric < self.best_loss:
            self.best_loss = comparison_metric
            self.counter = 0
            self.max_model = model.save_checkpoint()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
