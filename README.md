# Early CDR-Based Dementia Classification from Structural MRI

This repository contains the code used for the MRP investigating 2D and 3D transfer learning approaches for CDR-based dementia classification from structural MRI.

The project uses the OASIS-1 dataset to classify subjects as cognitively normal (CDR = 0) or CDR-positive (CDR >= 0.5). Three deep learning models were evaluated:

- VGG16 – 2D slice-based transfer learning
- ResNet50 – 2D slice-based transfer learning
- MedicalNet-pretrained ResNet18 – 3D volumetric transfer learning

Subject-level predictions from the three models were also combined using an equal-weight probability ensemble.

## Dataset

This project uses the OASIS-1 Cross-Sectional dataset from the Open Access Series of Imaging Studies (OASIS). The MRI data are not included in this repository and must be obtained separately through OASIS.

### OASIS Acknowledgement

Data were provided by OASIS-1: Cross-Sectional. Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, and J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, and U24 RR021382.

### Citation

Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. Journal of Cognitive Neuroscience, 19(9), 1498–1507.

## Preprocessing

MRI preprocessing included:

- Skull stripping using SynthStrip
- Z-score intensity normalization
- Extraction of 31 central coronal slices for the 2D models
- Resizing 2D slices to 224 x 224
- Resizing 3D volumes to 128 x 128 x 128
- Subject-level train/validation/test splitting

## Models

### VGG16
ImageNet-pretrained VGG16 was fine-tuned for binary classification using 2D coronal MRI slices.

### ResNet50
ImageNet-pretrained ResNet50 was fine-tuned for binary classification using 2D coronal MRI slices.

### MedicalNet
A MedicalNet-pretrained 3D ResNet18 was used for volumetric MRI classification. The MedicalNet implementation used by the project is included in the repository.

## Evaluation

Models were evaluated at the subject level using:

- Accuracy
- Precision
- Recall
- Specificity
- ROC-AUC
- Confusion matrices

Grad-CAM was also used to qualitatively examine regions contributing to model predictions.

## Requirements

The project was developed using Python 3.11.

Main dependencies include:

- TensorFlow 2.20.0
- PyTorch 2.13.0
- TorchIO 1.2.1
- torchinfo 1.8.0
- NumPy 2.4.6
- pandas 3.0.5
- SciPy 1.17.1
- scikit-learn 1.9.0
- Matplotlib 3.11.1
- NiBabel 5.4.2

See `requirements.txt` for the Python package requirements.

## Notes

The OASIS MRI data are not distributed with this repository due to data-access requirements.

Saved model files may be stored using Git LFS due to their file size.
