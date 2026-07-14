from mrp_models import vgg16, resnet50, cnn_3d
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import torch
from torch.utils.data import Dataset, DataLoader


LEARNING_RATE = 1e-4
BATCH_SIZE = 8
DROPOUT = 0.5
EPOCHS = 20

class DataLoader2D:
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data_2d(self, folder_path):
        images = []
        labels = []

        for diag_class, label in [("normal", 0), ("diseased", 1)]:
            class_path = folder_path + "/" + diag_class

            for subject_id in os.listdir(class_path):
                subject_path = class_path + "/" + subject_id

                for file in sorted(os.listdir(subject_path)):
                    if not file.endswith(".npy"):
                        continue

                    img = np.load(subject_path + "/" + file).astype(np.float32)

                    img = np.expand_dims(img, axis=-1)   # (224, 224, 1)
                    img = np.repeat(img, 3, axis=-1)    # (224, 224, 3)

                    images.append(img)
                    labels.append(label)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        return tf.data.Dataset.from_tensor_slices((images, labels))

    def load_and_augment_2d(self, path_root, batch_size=8):
        self.train_data = self.load_data_2d(path_root + "/train/2d")
        self.val_data = self.load_data_2d(path_root + "/val/2d")
        self.test_data = self.load_data_2d(path_root + "/test/2d")

        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.02),
            layers.RandomZoom(0.1),
            layers.RandomFlip("horizontal")
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
    def __init__(self, folder_path):
        self.samples = []

        for diag_class, label in [("normal", 0), ("diseased", 1)]:
            class_path = folder_path + "/" + diag_class
            for file in sorted(os.listdir(class_path)):
                if file.endswith(".npy"):
                    self.samples.append((class_path + "/" + file, label))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        volume = np.load(path).astype(np.float32)   # (128,128,128,1)
        volume = np.transpose(volume, (3, 0, 1, 2)) # (1,128,128,128)
        volume = torch.from_numpy(volume)
        label = torch.tensor(label, dtype=torch.float32)

        return volume, label
    

def load_data_3d(path_root):

    train_dataset = DataLoader3D(path_root + "/train/3d")
    val_dataset = DataLoader3D(path_root + "/val/3d")
    test_dataset = DataLoader3D(path_root + "/test/3d")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    loader2d = DataLoader2D()
    train_data, val_data, test_data = loader2d.load_and_augment_2d("/Volumes/T9/processed")

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
    train_loader3d, val_loader3d, test_loader3d = load_data_3d("/Volumes/T9/processed")

    medicalnet = cnn_3d.MedicalNet()
    medicalnet.build_model()

    medicalnet.train_model(train_loader3d, val_loader3d, epochs=EPOCHS, learning_rate=LEARNING_RATE)   
    medicalnet_results = medicalnet.evaluate_model(test_loader3d)

    print(medicalnet_results)