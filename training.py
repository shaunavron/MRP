from mrp_models import vgg16, resnet50, cnn_3d
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import torch
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)


LEARNING_RATE = 1e-6
BATCH_SIZE_2D = 8
BATCH_SIZE_3D = 2
DROPOUT = 0.5
EPOCHS = 20

def save_experiment(results):
    os.makedirs("results", exist_ok=True)
    path = "results/experiments.csv"
    file_exists = os.path.exists(path)
    
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

class DataLoader2D:
    def __init__(self):
        self.train_data = None
        self.train_subject_ids = None
        self.val_data = None
        self.val_subject_ids = None
        self.test_data = None
        self.test_subject_ids = None

    def load_data_2d(self, folder_path):
        images = []
        labels = []
        subject_ids = []

        for diag_class, label in [("normal", 0), ("diseased", 1)]:
            class_path = folder_path + "/" + diag_class

            for subject_id in os.listdir(class_path):
                if subject_id.startswith("._"):
                    continue
                subject_path = class_path + "/" + subject_id

                for file in sorted(os.listdir(subject_path)):
                    if not file.endswith(".npy") or file.startswith("._"):
                        continue

                    file_path = os.path.join(subject_path, file)

                    img = np.load(file_path).astype(np.float32)
                    img = np.expand_dims(img, axis=-1)   # (224, 224, 1)
                    img = np.repeat(img, 3, axis=-1)    # (224, 224, 3)

                    images.append(img)
                    labels.append(label)
                    subject_ids.append(subject_id)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        subject_ids=np.array(subject_ids)

        return tf.data.Dataset.from_tensor_slices((images, labels)), subject_ids

    def load_and_augment_2d(self, path_root, batch_size=BATCH_SIZE_2D):
        self.train_data, self.train_subject_ids = self.load_data_2d(path_root + "/train/2d")
        self.val_data, self.val_subject_ids = self.load_data_2d(path_root + "/val/2d")
        self.test_data, self.test_subject_ids = self.load_data_2d(path_root + "/test/2d")

        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.02),
            layers.RandomZoom(0.1)
        ])

        self.train_data = self.train_data.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)

        self.train_data = self.train_data.map(lambda x, y: (data_augmentation(tf.cast(x, tf.float32),training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

        self.val_data = self.val_data.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

        self.test_data = self.test_data.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

        self.train_data = self.train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.val_data = self.val_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.test_data = self.test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return self.train_data, self.val_data, self.test_data


class DataLoader3D(Dataset):
    def __init__(self, folder_path, augment=False):
        self.samples = []
        self.augment = augment

        self.transform = tio.Compose([
            tio.RandomAffine(
                scales=(0.98, 1.02),
                degrees=2,
                translation=2,
                p=0.3
            )
        ])

        for diag_class, label in [("normal", 0), ("diseased", 1)]:
            class_path = folder_path + "/" + diag_class
            for file in sorted(os.listdir(class_path)):
                if file.endswith(".npy") and not file.startswith("._"):
                    self.samples.append((class_path + "/" + file, label))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        volume = np.load(path).astype(np.float32)   # (128,128,128,1)
        volume = np.transpose(volume, (3, 0, 1, 2)) # (1,128,128,128)
        volume = torch.from_numpy(volume)

        if self.augment:
            volume = self.transform(volume)

        label = torch.tensor(label, dtype=torch.float32)

        return volume, label
    

def load_data_3d(path_root, batch_size=BATCH_SIZE_3D):

    train_dataset = DataLoader3D(path_root + "/train/3d", augment=True)
    val_dataset = DataLoader3D(path_root + "/val/3d")
    test_dataset = DataLoader3D(path_root + "/test/3d")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    loader2d = DataLoader2D()
    train_data, val_data, test_data = loader2d.load_and_augment_2d("/Volumes/T9/processed_skullstripped")

    # VGG16
    vgg = vgg16.VGG16Model()
    vgg.build_model(learning_rate=LEARNING_RATE, dropout=DROPOUT)
    vgg.train_model(train_data, val_data, epochs=EPOCHS, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

    vgg_results = vgg.evaluate_model(test_data)

    # ResNet50
    resnet = resnet50.ResNet50Model()
    resnet.build_model(learning_rate=LEARNING_RATE, dropout=DROPOUT)
    resnet.train_model(train_data, val_data, epochs=EPOCHS, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    
    resnet_results = resnet.evaluate_model(test_data)
    
    # MedicalNet
    train_loader3d, val_loader3d, test_loader3d = load_data_3d("/Volumes/T9/processed_skullstripped")

    medicalnet = cnn_3d.MedicalNet()
    medicalnet.build_model()

    checkpoint_path = (f"saved_models/medicalnet_training_hist.pth")

    medicalnet.train_model(train_loader3d, val_loader3d, epochs=EPOCHS, learning_rate=LEARNING_RATE, checkpoint_path=checkpoint_path)   

    checkpoint = torch.load(checkpoint_path, map_location=medicalnet.device)

    medicalnet.model.load_state_dict(checkpoint["model_state_dict"])

    test_results = medicalnet.evaluate_model(test_loader3d)

    results = {
        "model": "MedicalNet",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE_3D,
        "epochs": EPOCHS,
        "dropout": 0.0,
        "unfrozen_layers": "layer4 + classification_head",
        "best_epoch": checkpoint["epoch"],
        "val_loss": checkpoint["val_loss"],
        "val_accuracy": checkpoint["val_accuracy"],
        "val_precision": checkpoint["val_precision"],
        "val_recall": checkpoint["val_recall"],
        "val_auc": checkpoint["val_auc"]
        }

    save_experiment(results)
