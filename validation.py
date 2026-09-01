import os
import csv
import tensorflow as tf
import torch

from torch.utils.data import DataLoader
from mrp_models import cnn_3d, vgg16, resnet50
from training import DataLoader2D, DataLoader3D


BATCH_SIZE_2D = 8
BATCH_SIZE_3D = 2

# 2D

loader2d = DataLoader2D()

oasis1_val_2d, _ = loader2d.load_data_2d("/Volumes/T9/processed_skullstripped/val/2d")
oasis1_val_2d = (oasis1_val_2d.batch(BATCH_SIZE_2D).prefetch(tf.data.AUTOTUNE))

oasis1_test_2d, _ = loader2d.load_data_2d("/Volumes/T9/processed_skullstripped/test/2d")
oasis1_test_2d = (oasis1_test_2d.batch(BATCH_SIZE_2D).prefetch(tf.data.AUTOTUNE))

# 3D

oasis1_val_3d = DataLoader3D("/Volumes/T9/processed_skullstripped/val/3d", augment=False)
oasis1_val_3d_loader = DataLoader(oasis1_val_3d, batch_size=BATCH_SIZE_3D, shuffle=False)

oasis1_test_3d = DataLoader3D("/Volumes/T9/processed_skullstripped/test/3d", augment=False)
oasis1_test_3d_loader = DataLoader(oasis1_test_3d, batch_size=BATCH_SIZE_3D, shuffle=False)


results = []

vgg = vgg16.VGG16Model()
vgg.build_model(
    learning_rate=1e-5,
    dropout=0.3
)

vgg.model.load_weights(
    "saved_models/vgg_best.keras"
)

print("\nEvaluating VGG16...")

vgg_val = vgg.model.evaluate(oasis1_val_2d, return_dict=True)

results.append({
    "model": "VGG16",
    "dataset": "OASIS-1",
    "split": "validation",
    **vgg_val
})

vgg_test = vgg.model.evaluate(oasis1_test_2d, return_dict=True)

results.append({
    "model": "VGG16",
    "dataset": "OASIS-1",
    "split": "test",
    **vgg_test
})

resnet = resnet50.ResNet50Model()
resnet.build_model(
    learning_rate=1e-5,
    dropout=0.7
)

resnet.model.load_weights(
    "saved_models/resnet_best.keras"
)

print("\nEvaluating ResNet50...")

resnet_val = resnet.model.evaluate(oasis1_val_2d, return_dict=True)

results.append({
    "model": "ResNet50",
    "dataset": "OASIS-1",
    "split": "validation",
    **resnet_val
})

resnet_test = resnet.model.evaluate(oasis1_test_2d, return_dict=True)

results.append({
    "model": "ResNet50",
    "dataset": "OASIS-1",
    "split": "test",
    **resnet_test
})

medicalnet = cnn_3d.MedicalNet()
medicalnet.build_model()

checkpoint_path = (
    "saved_models/"
    "medicalnet_lr1e-06_bs2.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=medicalnet.device
)

medicalnet.model.load_state_dict(
    checkpoint["model_state_dict"]
)

medicalnet.model.eval()

print("\nEvaluating MedicalNet...")

medicalnet_val = medicalnet.evaluate_model(
    oasis1_val_3d_loader
)

medicalnet_test = medicalnet.evaluate_model(
    oasis1_test_3d_loader
)

results.extend([
    {
        "model": "MedicalNet",
        "dataset": "OASIS-1",
        "split": "validation",
        "loss": medicalnet_val["test_loss"],
        "accuracy": medicalnet_val["test_accuracy"],
        "precision": medicalnet_val["test_precision"],
        "recall": medicalnet_val["test_recall"],
        "auc": medicalnet_val["test_auc"],
    },
    {
        "model": "MedicalNet",
        "dataset": "OASIS-1",
        "split": "test",
        "loss": medicalnet_test["test_loss"],
        "accuracy": medicalnet_test["test_accuracy"],
        "precision": medicalnet_test["test_precision"],
        "recall": medicalnet_test["test_recall"],
        "auc": medicalnet_test["test_auc"],
    }
])

os.makedirs(
    "results",
    exist_ok=True
)

output_path = (
    "results/"
    "final_evaluation.csv"
)

fieldnames = []

for row in results:
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)

with open(
    output_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


print(
    f"\nResults saved to "
    f"{output_path}"
)
