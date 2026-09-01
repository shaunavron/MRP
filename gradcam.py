import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mrp_models import cnn_3d
from training import DataLoader3D
from torchinfo import summary

### 2D ###

vgg_model = tf.keras.models.load_model(
    "/Users/shauna/MRP/saved_models/vgg_best.keras"
)

vgg_model.summary()

resnet_model = tf.keras.models.load_model(
    "/Users/shauna/MRP/saved_models/resnet_best.keras"
)

resnet_model.summary()

def make_vgg_gradcam(img_array, model):
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer("block5_conv3")

    # Model from image -> block5_conv3
    conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # Classification head after the VGG backbone
    classifier_layers = model.layers[1:]

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        x = conv_output
        x = base_model.get_layer("block5_pool")(x)

        for layer in classifier_layers:
            x = layer(x, training=False)
        prediction = x[:, 0]

    grads = tape.gradient(prediction, conv_output)
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy(), float(prediction[0])

def make_resnet_gradcam(img_array, model):
    base_model = model.layers[0]
    target_layer = base_model.get_layer("conv5_block3_out")

    conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=target_layer.output
    )

    # Classification head after the ResNet backbone
    classifier_layers = model.layers[1:]
    
    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        x = conv_output
        x = base_model.get_layer("conv5_block3_out")(x)

        for layer in classifier_layers:
            x = layer(x, training=False)
        prediction = x[:, 0]

    grads = tape.gradient(prediction, conv_output)
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy(), float(prediction[0])

img = np.load("/Volumes/T9/processed_skullstripped/test/2d/normal/OAS1_0070_MR1/slice_104.npy").astype(np.float32)

def prepare_img(img):
    img_rgb = np.expand_dims(img, axis=-1)
    img_rgb = np.repeat(img_rgb, 3, axis=-1)
    return np.expand_dims(img_rgb, axis=0)

img = np.load(
    "/Volumes/T9/processed_skullstripped/test/2d/normal/OAS1_0070_MR1/slice_104.npy"
).astype(np.float32)

img_input = prepare_img(img)

vgg_heatmap, vgg_prob = make_vgg_gradcam(
    img_input,
    vgg_model
)

resnet_heatmap, resnet_prob = make_resnet_gradcam(
    img_input,
    resnet_model
)

vgg_heatmap = tf.image.resize(
    vgg_heatmap[..., np.newaxis],
    img.shape
).numpy().squeeze()

resnet_heatmap = tf.image.resize(
    resnet_heatmap[..., np.newaxis],
    img.shape
).numpy().squeeze()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original MRI")
axes[0].axis("off")

axes[1].imshow(img, cmap="gray")
axes[1].imshow(
    vgg_heatmap,
    cmap="jet",
    alpha=0.4
)
axes[1].set_title(
    f"VGG16 Grad-CAM\nP(diseased)={vgg_prob:.3f}"
)
axes[1].axis("off")

axes[2].imshow(img, cmap="gray")
axes[2].imshow(
    resnet_heatmap,
    cmap="jet",
    alpha=0.4
)
axes[2].set_title(
    f"ResNet50 Grad-CAM\nP(diseased)={resnet_prob:.3f}"
)
axes[2].axis("off")

fig.suptitle(
    "Grad-CAM Visualisation – CDR 0",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

### 3D ###

medicalnet = cnn_3d.MedicalNet()

medicalnet.build_model()

checkpoint = torch.load(
    "/Users/shauna/MRP/saved_models/medicalnet_lr1e-06_bs2.pth",
    map_location=medicalnet.device
)

medicalnet.model.load_state_dict(
    checkpoint["model_state_dict"]
)

medicalnet.model.eval()

summary(

    medicalnet.model,

    input_size=(1, 1, 128, 128, 128),

    col_names=("output_size", "num_params", "trainable"),

    depth=2

)

# 3D grad cam 

def make_medicalnet_gradcam(input_tensor, model):

    target_layer = model.layer4[-1]

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    model.zero_grad()

    output = model(input_tensor).view(-1)

    # Single binary-classification logit.
    # Positive direction corresponds to diseased class.
    score = output[0]

    score.backward()

    activation = activations["value"]
    gradient = gradients["value"]

    # Average gradient across D, H, W
    weights = gradient.mean(
        dim=(2, 3, 4),
        keepdim=True
    )

    # Weighted sum of feature maps
    heatmap = torch.sum(
        weights * activation,
        dim=1,
        keepdim=True
    )

    heatmap = torch.relu(heatmap)

    # Resize from layer4 resolution back to 128 x 128 x 128
    heatmap = F.interpolate(
        heatmap,
        size=input_tensor.shape[2:],
        mode="trilinear",
        align_corners=False
    )

    heatmap = heatmap[0, 0]

    # Normalize to 0-1
    heatmap = heatmap - heatmap.min()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    probability = torch.sigmoid(
        output[0]
    ).item()

    heatmap = (
        heatmap
        .detach()
        .cpu()
        .numpy()
    )

    forward_handle.remove()
    backward_handle.remove()

    return heatmap, probability


volume_np = np.load("/Volumes/T9/processed_skullstripped/test/3d/normal/OAS1_0070_MR1.npy").astype(np.float32)
print("Loaded shape:", volume_np.shape)

volume_np = np.transpose(
    volume_np,
    (3, 0, 1, 2)
)

input_tensor = torch.tensor(
    volume_np,
    dtype=torch.float32
).unsqueeze(0)

input_tensor = input_tensor.to(
    medicalnet.device
)

print("Model input shape:", input_tensor.shape)

heatmap_3d, med_prob = make_medicalnet_gradcam(
    input_tensor,
    medicalnet.model
)

print("Heatmap shape:", heatmap_3d.shape)
print("P(diseased):", med_prob)


volume_np = (
    input_tensor[0, 0]
    .detach()
    .cpu()
    .numpy()
)

x = volume_np.shape[0] // 2
y = volume_np.shape[1] // 2
z = volume_np.shape[2] // 2

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

# original mri 

axes[0, 0].imshow(
    volume_np[x, :, :].T,
    cmap="gray",
    origin="lower"
)
axes[0, 0].set_title("View 1")

axes[0, 1].imshow(
    volume_np[:, y, :].T,
    cmap="gray",
    origin="lower"
)
axes[0, 1].set_title("View 2")

axes[0, 2].imshow(
    volume_np[:, :, z].T,
    cmap="gray",
    origin="lower"
)
axes[0, 2].set_title("View 3")

# After grad cam

axes[1, 0].imshow(
    volume_np[x, :, :].T,
    cmap="gray",
    origin="lower"
)
axes[1, 0].imshow(
    heatmap_3d[x, :, :].T,
    cmap="jet",
    alpha=0.4,
    origin="lower"
)

axes[1, 1].imshow(
    volume_np[:, y, :].T,
    cmap="gray",
    origin="lower"
)
axes[1, 1].imshow(
    heatmap_3d[:, y, :].T,
    cmap="jet",
    alpha=0.4,
    origin="lower"
)

axes[1, 2].imshow(
    volume_np[:, :, z].T,
    cmap="gray",
    origin="lower"
)
axes[1, 2].imshow(
    heatmap_3d[:, :, z].T,
    cmap="jet",
    alpha=0.4,
    origin="lower"
)

for ax in axes.flat:
    ax.axis("off")

plt.suptitle(
    f"MedicalNet 3D Grad-CAM - CDR 0\n"
    f"P(diseased) = {med_prob:.3f}"
)

plt.tight_layout()
plt.show()
