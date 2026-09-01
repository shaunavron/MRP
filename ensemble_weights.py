import os
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


VAL_2D_ROOT = "/Volumes/T9/processed_skullstripped/val/2d"
VAL_3D_ROOT = "/Volumes/T9/processed_skullstripped/val/3d"

VGG_CHECKPOINT = "/Users/shauna/MRP/saved_models/vgg_best.keras"
RESNET_CHECKPOINT = "/Users/shauna/MRP/saved_models/resnet_best.keras"
MEDNET_CHECKPOINT = "/Users/shauna/MRP/saved_models/medicalnet_lr1e-06_bs2.pth"


vgg = vgg16.VGG16Model()
vgg.build_model(
    learning_rate=1e-5,
    dropout=0.3
)
vgg.model.load_weights(VGG_CHECKPOINT)


resnet = resnet50.ResNet50Model()
resnet.build_model(
    learning_rate=1e-5,
    dropout=0.7
)
resnet.model.load_weights(RESNET_CHECKPOINT)


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


def prepare_2d_slice(img):

    img = np.expand_dims(
        img,
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
        raise ValueError(
            f"{subject_dir} has "
            f"{len(slice_files)} slices"
        )

    images = []

    for filename in slice_files:

        img = np.load(
            os.path.join(
                subject_dir,
                filename
            )
        ).astype(np.float32)

        images.append(
            prepare_2d_slice(img)
        )

    images = np.stack(
        images,
        axis=0
    )

    probabilities = model.predict(
        images,
        batch_size=8,
        verbose=0
    ).reshape(-1)

    return float(
        np.mean(probabilities)
    )


def predict_medicalnet(volume_path):

    volume = np.load(
        volume_path
    ).astype(np.float32)

    # (128,128,128,1) -> (1,128,128,128)

    volume = np.transpose(
        volume,
        (3, 0, 1, 2)
    )

    # (1,1,128,128,128)

    volume = torch.tensor(
        volume,
        dtype=torch.float32
    ).unsqueeze(0)

    volume = volume.to(
        medicalnet.device
    )

    with torch.no_grad():

        logit = medicalnet.model(
            volume
        ).view(-1)[0]

        probability = torch.sigmoid(
            logit
        ).item()

    return probability


def metrics_from_probs(
    y_true,
    probabilities
):

    y_true = np.array(
        y_true,
        dtype=int
    )

    probabilities = np.array(
        probabilities,
        dtype=float
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

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
        "auc": auc
    }

# GENERATE VALIDATION SUBJECT PROBABILITIES

validation_subjects = []

class_info = {
    "normal": 0,
    "diseased": 1
}


for class_name, true_label in class_info.items():

    class_dir = os.path.join(
        VAL_2D_ROOT,
        class_name
    )

    subjects = sorted([
        subject
        for subject in os.listdir(
            class_dir
        )
        if os.path.isdir(
            os.path.join(
                class_dir,
                subject
            )
        )
    ])

    for subject_id in subjects:

        print(
            f"Processing {subject_id}"
        )

        subject_2d_dir = os.path.join(
            VAL_2D_ROOT,
            class_name,
            subject_id
        )

        vgg_prob = predict_2d_subject(
            subject_2d_dir,
            vgg.model
        )

        resnet_prob = predict_2d_subject(
            subject_2d_dir,
            resnet.model
        )

        volume_path = os.path.join(
            VAL_3D_ROOT,
            class_name,
            f"{subject_id}.npy"
        )

        med_prob = predict_medicalnet(
            volume_path
        )

        validation_subjects.append({
            "subject_id": subject_id,
            "true_label": true_label,
            "vgg": vgg_prob,
            "resnet": resnet_prob,
            "medicalnet": med_prob
        })

# FIXED CANDIDATE WEIGHTS

weight_sets = [
    {
        "name": "equal",
        "vgg": 1/3,
        "resnet": 1/3,
        "medicalnet": 1/3
    },
    {
        "name": "40_20_40",
        "vgg": 0.40,
        "resnet": 0.20,
        "medicalnet": 0.40
    },
    {
        "name": "40_10_50",
        "vgg": 0.40,
        "resnet": 0.10,
        "medicalnet": 0.50
    },
    {
        "name": "45_10_45",
        "vgg": 0.45,
        "resnet": 0.10,
        "medicalnet": 0.45
    },
    {
        "name": "35_15_50",
        "vgg": 0.35,
        "resnet": 0.15,
        "medicalnet": 0.50
    }
]

# EVALUATE WEIGHTS ON VALIDATION ONLY

y_true = [
    row["true_label"]
    for row in validation_subjects
]

weight_results = []


for weights in weight_sets:

    probs = []

    for row in validation_subjects:

        ensemble_prob = (
            weights["vgg"]
            * row["vgg"]

            + weights["resnet"]
            * row["resnet"]

            + weights["medicalnet"]
            * row["medicalnet"]
        )

        probs.append(
            ensemble_prob
        )

    metrics = metrics_from_probs(
        y_true,
        probs
    )

    result = {
        **weights,
        **metrics
    }

    weight_results.append(
        result
    )


# ============================================================
# PRINT ALL RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("VALIDATION ENSEMBLE WEIGHT COMPARISON")
print("=" * 80)

for row in weight_results:

    print(
        f"\n{row['name']}"
    )

    print(
        f"Weights: "
        f"VGG={row['vgg']:.2f}, "
        f"ResNet={row['resnet']:.2f}, "
        f"MedicalNet={row['medicalnet']:.2f}"
    )

    print(
        f"AUC:         "
        f"{row['auc']:.4f}"
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


# ============================================================
# SELECT BEST
# Primary: validation AUC
# Tie-breaker: validation accuracy
# ============================================================

best = max(
    weight_results,
    key=lambda x: (
        x["auc"],
        x["accuracy"]
    )
)

print("\n")
print("=" * 80)
print("SELECTED WEIGHTS")
print("=" * 80)

print(
    f"Scheme: {best['name']}"
)

print(
    f"VGG16:      {best['vgg']:.2f}"
)

print(
    f"ResNet50:   {best['resnet']:.2f}"
)

print(
    f"MedicalNet: {best['medicalnet']:.2f}"
)

print(
    f"Validation AUC: "
    f"{best['auc']:.4f}"
)

print(
    f"Validation accuracy: "
    f"{best['accuracy']:.4f}"
)
