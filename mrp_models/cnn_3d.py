import torch
import torch.nn as nn
from MedicalNet.models import resnet
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

class MedicalNet:
    def __init__(self):
        self.model = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_auc": [],
            "val_auc": [],
            "train_precision": [],
            "val_precision": [],
            "train_recall": [],
            "val_recall": []
        }
    
    def build_model(self):

        self.model = resnet.resnet18(
            sample_input_W=128,
            sample_input_H=128,
            sample_input_D=128,
            shortcut_type="A",
            no_cuda=True,
            num_seg_classes=1
        )

        checkpoint = torch.load(
            "/Users/shauna/MRP/MedicalNet/pretrain/resnet_18_23dataset.pth",
            map_location="cpu"
        )

        self.model.load_state_dict(
            checkpoint["state_dict"],
            strict=False
        )

        # Replace segmentation head
        self.model.conv_seg = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

        # Layer freezing and unfreezing
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        for param in self.model.conv_seg.parameters():
            param.requires_grad = True

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model.to(self.device)

    def train_model(self, train_loader, val_loader, epochs, learning_rate):
        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )

        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = 0.0

            for volumes, labels in train_loader:
                volumes = volumes.to(self.device)
                labels = labels.float().to(self.device).view(-1)

                optimizer.zero_grad()

                outputs = self.model(volumes).view(-1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0

            all_labels = []
            all_predictions = []
            all_probabilities = []

            with torch.no_grad():
                for volumes, labels in val_loader:
                    volumes = volumes.to(self.device)
                    labels = labels.float().to(self.device).view(-1)

                    outputs = self.model(volumes).view(-1)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities >= 0.5).float()

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

            val_loss /= len(val_loader)

            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(
                all_labels,
                all_predictions,
                zero_division=0
            )
            recall = recall_score(
                all_labels,
                all_predictions,
                zero_division=0
            )
            auc = roc_auc_score(all_labels, all_probabilities)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(accuracy)
            self.history["val_precision"].append(precision)
            self.history["val_recall"].append(recall)
            self.history["val_auc"].append(auc)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"train_loss: {train_loss:.4f} - "
                f"val_loss: {val_loss:.4f} - "
                f"val_accuracy: {accuracy:.4f} - "
                f"val_auc: {auc:.4f}"
            )

    def evaluate_model(self, test_loader):
        criterion = nn.BCEWithLogitsLoss()

        self.model.eval()

        test_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for volumes, labels in test_loader:
                volumes = volumes.to(self.device)
                labels = labels.float().to(self.device).view(-1)

                outputs = self.model(volumes).view(-1)

                loss = criterion(outputs, labels)
                test_loss += loss.item()

                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities >= 0.5).float()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        test_loss /= len(test_loader)

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(
            all_labels,
            all_predictions,
            zero_division=0
        )
        recall = recall_score(
            all_labels,
            all_predictions,
            zero_division=0
        )
        auc = roc_auc_score(all_labels, all_probabilities)

        return {
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_auc": auc,
            "labels": all_labels,
            "predictions": all_predictions,
            "probabilities": all_probabilities
        }