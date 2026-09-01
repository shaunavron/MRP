import os
import csv
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
from mrp_models import vgg16, resnet50, cnn_3d


TEST_2D_ROOT = "/Volumes/T9/processed_skullstripped/test/2d"
TEST_3D_ROOT = "/Volumes/T9/processed_skullstripped/test/3d"

VGG_CHECKPOINT = "/Users/shauna/MRP/saved_models/vgg_best.keras"
RESNET_CHECKPOINT = "/Users/shauna/MRP/saved_models/resnet_best.keras"
MEDNET_CHECKPOINT = "/Users/shauna/MRP/saved_models/medicalnet_lr1e-06_bs2.pth"

OUTPUT_DIR = "/Users/shauna/MRP/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


vgg = vgg16.VGG16Model()
vgg.build_model(
    learning_rate=1e-5,
    dropout=0.3
)

vgg.model.load_weights(VGG_CHECKPOINT)
vgg.model.trainable = False

resnet = resnet50.ResNet50Model()
resnet.build_model(
    learning_rate=1e-5,
    dropout=0.7
)

resnet.model.load_weights(RESNET_CHECKPOINT)
resnet.model.trainable = False

medicalnet = cnn_3d.MedicalNet()
medicalnet.build_model()

checkpoint = torch.load(
    MEDNET_CHECKPOINT,
    map_location=medicalnet.device
)

medicalnet.model.load_state_dict(
    checkpoint["model_state_dict"]
)

medicalnet.model.eval()

def prepare_2d_slice(slice_img):
    # (224, 224)->(224, 224, 3)
    img = np.expand_dims(
        slice_img,
        axis=-1
    )
    img = np.repeat(
        img,
        3,
        axis=-1
    )
    return img.astype(np.float32)


def predict_2d_subject(subject_dir, model):
    slice_files = sorted([
        f
        for f in os.listdir(subject_dir)
        if f.endswith(".npy")
        and not f.startswith("._")
    ])

    if len(slice_files) != 31:
        print(
            f"WARNING: {subject_dir} "
            f"contains {len(slice_files)} slices"
        )

    images = []

    for filename in slice_files:
        path = os.path.join(
            subject_dir,
            filename
        )
        img = np.load(path).astype(np.float32)
        img = prepare_2d_slice(img)
        images.append(img)

    images = np.stack(images,axis=0)

    probabilities = model.predict(
        images,
        batch_size=8,
        verbose=0
    ).reshape(-1)

    subject_probability = float(np.mean(probabilities))

    return (subject_probability, probabilities)

def predict_medicalnet_subject(volume_path,medicalnet):
    volume = np.load(
        volume_path
    ).astype(np.float32)

    # (128, 128, 128, 1)
    if volume.shape != (
        128,
        128,
        128,
        1
    ):
        raise ValueError(
            f"Unexpected 3D shape: "
            f"{volume.shape} "
            f"for {volume_path}"
        )

    # (1, 128, 128, 128)
    volume = np.transpose(
        volume,
        (3, 0, 1, 2)
    )

    # (1, 1, 128, 128, 128)
    volume = torch.tensor(
        volume,
        dtype=torch.float32
    ).unsqueeze(0)

    volume = volume.to(medicalnet.device)

    with torch.no_grad():
        logit = medicalnet.model(
            volume
        ).view(-1)[0]

        probability = torch.sigmoid(
            logit
        ).item()

    return probability


def calculate_metrics(y_true, y_prob, threshold=0.5):

    y_true = np.array(y_true, dtype=int)

    y_prob = np.array(y_prob,dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true,y_pred)

    precision = precision_score(y_true, y_pred, zero_division=0)

    recall = recall_score(y_true, y_pred, zero_division=0)

    auc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# Subject level pred

subject_results = []

class_info = {
    "normal": 0,
    "diseased": 1
}

for class_name, true_label in class_info.items():
    # 2D
    class_2d_dir = os.path.join(TEST_2D_ROOT, class_name)

    subject_ids = sorted([
        name
        for name in os.listdir(
            class_2d_dir
        )
        if os.path.isdir(
            os.path.join(
                class_2d_dir,
                name
            )
        )
    ])

    for subject_id in subject_ids:
        print(
            f"Processing "
            f"{subject_id}..."
        )

        subject_2d_dir = os.path.join(TEST_2D_ROOT, class_name, subject_id)

        (vgg_probability, vgg_slice_probs) = predict_2d_subject(subject_2d_dir, vgg.model)

        (resnet_probability, resnet_slice_probs) = predict_2d_subject(subject_2d_dir, resnet.model)

        # 3D
        volume_path = os.path.join(TEST_3D_ROOT, class_name, f"{subject_id}.npy")

        if not os.path.exists(volume_path):
            raise FileNotFoundError(
                f"Missing 3D volume: "
                f"{volume_path}"
            )

        medicalnet_probability = (predict_medicalnet_subject(volume_path, medicalnet))

        # Ensemble
        ensemble_probability = float(
            np.mean([
                vgg_probability,
                medicalnet_probability
            ])
        )

        subject_results.append({
            "subject_id":
                subject_id,

            "true_label":
                true_label,

            "vgg_probability":
                vgg_probability,

            "resnet_probability":
                resnet_probability,

            "medicalnet_probability":
                medicalnet_probability,

            "ensemble_probability":
                ensemble_probability,

            "vgg_prediction":
                int(
                    vgg_probability >= 0.5
                ),

            "resnet_prediction":
                int(
                    resnet_probability >= 0.5
                ),

            "medicalnet_prediction":
                int(
                    medicalnet_probability >= 0.5
                ),

            "ensemble_prediction":
                int(
                    ensemble_probability >= 0.5
                )
        })

subject_output = os.path.join(
    OUTPUT_DIR,
    "subject_level_predictions.csv"
)

with open(
    subject_output,
    "w",
    newline=""
) as f:

    fieldnames = list(
        subject_results[0].keys()
    )

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(
        subject_results
    )

print(
    f"\nSubject predictions saved to "
    f"{subject_output}"
)

y_true = [
    row["true_label"]
    for row in subject_results
]

model_probabilities = {
    "VGG16": [
        row["vgg_probability"]
        for row in subject_results
    ],

    "ResNet50": [
        row["resnet_probability"]
        for row in subject_results
    ],

    "MedicalNet": [
        row["medicalnet_probability"]
        for row in subject_results
    ],

    "Ensemble": [
        row["ensemble_probability"]
        for row in subject_results
    ]
}


summary_results = []

for model_name, probabilities in (
    model_probabilities.items()
):

    metrics = calculate_metrics(
        y_true,
        probabilities
    )

    summary_results.append({
        "model": model_name,
        **metrics
    })

print("\n")
print("=" * 70)
print("FINAL SUBJECT-LEVEL TEST RESULTS")
print("=" * 70)

for row in summary_results:

    print(
        f"\n{row['model']}"
    )

    print(
        f"Accuracy:    "
        f"{row['accuracy']:.4f}"
    )

    print(
        f"Precision:   "
        f"{row['precision']:.4f}"
    )

    print(
        f"Recall:      "
        f"{row['recall']:.4f}"
    )

    print(
        f"Specificity: "
        f"{row['specificity']:.4f}"
    )

    print(
        f"AUC:         "
        f"{row['auc']:.4f}"
    )

    print(
        "Confusion matrix: "
        f"TN={row['tn']}, "
        f"FP={row['fp']}, "
        f"FN={row['fn']}, "
        f"TP={row['tp']}"
    )

summary_output = os.path.join(
    OUTPUT_DIR,
    "subject_level_metrics.csv"
)

with open(
    summary_output,
    "w",
    newline=""
) as f:

    fieldnames = list(
        summary_results[0].keys()
    )

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        summary_results
    )

print(
    f"\nMetrics saved to "
    f"{summary_output}"
)
