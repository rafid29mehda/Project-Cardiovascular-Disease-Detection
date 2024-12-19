Kaggle Inspired Code Link: https://www.kaggle.com/code/bonhart/brain-mri-data-visualization-unet-fpn#Image-segmentation-using-Vanilla-UNet,-UNet-with-ResNeXt50-backbone,-and-Feature-Pyramid-Network.

### Key Concepts and Flow:
1. **Data Import**:
   - **Kaggle Dataset Import**: The dataset `mateuszbuda/lgg-mri-segmentation` is downloaded using the `kagglehub` package. The dataset path is stored in `mateuszbuda_lgg_mri_segmentation_path`.
   - **Print Statement**: Confirms the data source is successfully imported.

2. **Step Overview**: 
   - Mentions the overall structure of the notebook, outlining the main steps involved in the project: data preparation, visualization, augmentations, model training, etc.

3. **Libraries**:
   - Various libraries are imported, including:
     - `kagglehub` for Kaggle datasets,
     - `numpy`, `pandas` for data handling,
     - `cv2` (OpenCV) for image processing,
     - `matplotlib.pyplot` for data visualization, and others.

4. **Global Variables**:
   - **Data Path (`DATA_PATH`)**: Specifies where the data is located.
   - **File Length Parameters**: `BASE_LEN`, `END_IMG_LEN`, and `END_MASK_LEN` are used for sorting images and masks by their file names.
   - **Image Size (`IMG_SIZE`)**: Specifies the size of images to be used for model training, set to 512.

5. **Data Preparation**:
   - **Data Mapping**: The code scans the dataset directory and creates a list (`data_map`) that pairs image paths with their respective directories.
   - **DataFrame Creation**: A DataFrame (`df`) is created with columns `dirname` (directory names) and `path` (image paths).

6. **Sorting**:
   - Images and masks are sorted using the `BASE_LEN`, `END_IMG_LEN`, and `END_MASK_LEN` parameters to align corresponding image-mask pairs.

7. **Diagnosis Column**:
   - **Function (`positiv_negativ_diagnosis`)**: This function checks if a mask contains any positive values (indicating a positive diagnosis). It is applied to the mask paths to add a new column `diagnosis` to the DataFrame.

8. **Data Distribution (Plotting)**:
   - **Diagnosis Distribution**: A bar chart is generated to show the distribution of positive vs. negative diagnoses.
   - **Patient-wise Distribution**: Another bar chart visualizes the diagnosis distribution for each patient.

9. **Data Visualization**:
   - A subset of images is selected to visualize samples from both positive and negative diagnoses. These are resized to `IMG_SIZE` and displayed using `matplotlib` with a grid layout.

---

This part sets up the data for subsequent steps, focusing on preprocessing, sorting, and visualization of images along with their diagnosis. Let me know when you're ready to proceed with the next part, or if you want to ask questions about this one.


### Plan for Understanding and Analyzing the Code

1. **Imports & Setup**: 
    - We are working with libraries such as `cv2`, `matplotlib`, `torch`, and `albumentations` for loading, visualizing, and processing images and masks.
    - There's also a mention of using a UNet model for segmenting images (likely for tumor detection in MRI scans).

2. **Initial Image and Mask Display**:
    - The script loads some MRI images and their corresponding masks, visualizing them with `matplotlib`.
    - The dataset is filtered for positive diagnoses and visualized for a sample of 5 positive cases.

3. **Data Augmentation**:
    - Various transformations are applied to augment the dataset, such as random rotations, flipping, and color adjustments using `albumentations`.
    - This will help increase the variety of the dataset and prevent overfitting when training the model.

4. **DataLoader Setup**:
    - The dataset is split into training, validation, and test sets using `train_test_split` from `sklearn`.
    - A custom `BrainMriDataset` class is created to handle the loading of images and their associated masks, along with any necessary transformations (such as resizing and augmentations).

5. **Model Setup (UNet)**:
    - A U-Net model is defined for segmentation, which includes both downsampling (encoder) and upsampling (decoder) parts. 
    - The model includes convolutional blocks with ReLU activations, and concatenates corresponding downsampling and upsampling layers.
    - The output layer applies a `sigmoid` function, which is typical for binary classification tasks (like detecting tumor presence).

6. **Model Output**:
    - An example of passing random input through the model is shown to ensure it works correctly.

---

### Code Analysis

The main functionality here is setting up a deep learning pipeline for training a segmentation model on MRI data. Let's break it down further:

#### 1. **Dataset Processing**
   - `BrainMriDataset` handles the loading of MRI images and their corresponding masks, applies transformations, and returns them in a format compatible with PyTorch models.

#### 2. **Data Augmentation**
   - Various transformations, such as random crops, flips, rotations, and pixel transformations, are defined in `strong_transforms` and `transforms`.
   - Augmentations are designed to improve model robustness by introducing variability in the training data.

#### 3. **Training, Validation, and Test Splits**
   - The dataset is split using `train_test_split`, ensuring a balanced distribution of diagnoses (stratified by diagnosis).

#### 4. **UNet Model**
   - The `UNet` class defines the architecture of a U-Net, a popular choice for segmentation tasks. It employs downsampling (using `conv_down` layers) and upsampling (using `conv_up` layers), ensuring the original spatial dimensions are retained after the operations.

#### 5. **Visualization**
   - The `show_aug` function is used to visualize a grid of augmented images and masks to verify the transformations before training.

---

### Suggestions for the Next Steps:

1. **Training the Model**:
    - Next, you might want to implement the actual training loop where the model is trained using the training data (`train_dataloader`). You could define a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and a training loop to update the model's weights.

2. **Evaluation**:
    - You can also evaluate the model's performance on the validation and test sets using relevant metrics (e.g., Dice coefficient or Intersection over Union).

---

### Possible Next Steps and Queries:

**a.** Add a training loop with loss function and optimizer to start training the UNet model.  
**b.** Integrate evaluation metrics like the Dice coefficient to track model performance during training and testing.

Would you like to proceed with training the model, or should we focus on other aspects (e.g., visualization, evaluation)?


### Plan

The code provided includes several components related to building a deep learning model for image segmentation, utilizing feature pyramid networks (FPN), UNet architecture, and ResNeXt50 backbone. Below is the breakdown of the various sections:

#### 1. **FPN (Feature Pyramid Network)**
   - The FPN module is designed to handle different feature map resolutions at different stages of the neural network. It is commonly used in object detection, and here it is being used for segmentation.
   - The code includes a **bottom-up path** where different convolution blocks progressively reduce the resolution of the input, followed by a **top-down path** that upsamples the feature maps while maintaining spatial resolution.
   - The code uses the `SegmentationBlock` to upsample and concatenate these feature maps from different levels of the pyramid.

#### 2. **UNet + ResNeXt50 Backbone**
   - A UNet-like architecture with a ResNeXt50 backbone is utilized here. The backbone is pretrained (from torchvision's ResNeXt50).
   - The model includes encoder-decoder pairs with skip connections between corresponding layers in the encoder and decoder. This helps in preserving spatial details important for segmentation tasks.

#### 3. **Segmentation Quality Metrics**
   - The **Dice Coefficient** is implemented as a metric to evaluate the segmentation quality. This is used to measure the overlap between predicted and ground truth masks.

#### 4. **Segmentation Loss**
   - The **Dice Loss** is used as a loss function for training, which is commonly used in segmentation tasks because of its focus on maximizing overlap between predicted and actual masks.
   - **BCE-Dice Loss** combines binary cross-entropy loss and Dice loss to ensure better model performance in segmentation tasks.

### Key Code Elements

1. **ConvReluUpsample Class**: This class is a basic convolution block followed by a ReLU activation and optional upsampling (for increasing the spatial resolution of feature maps).

2. **SegmentationBlock Class**: Combines multiple `ConvReluUpsample` blocks for upsampling and encoding.

3. **FPN Class**: Implements the feature pyramid network with down-sampling, lateral connections, and smooth layers to merge feature maps from different scales.

4. **ResNeXtUNet Class**: Implements the UNet architecture with a ResNeXt50 backbone for segmentation. It uses skip connections from the encoder to the decoder.

5. **Dice Coefficient & Loss**: These functions calculate segmentation quality and loss using the Dice coefficient, which is vital for evaluating segmentation accuracy.

---

### Full Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function for double convolution
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class ConvReluUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.make_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = self.make_upsample(x)
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        blocks = [ConvReluUpsample(in_channels, out_channels, upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(ConvReluUpsample(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class FPN(nn.Module):
    def __init__(self, n_classes=1, pyramid_channels=256, segmentation_channels=256):
        super().__init__()
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)
        self.conv_down5 = double_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.seg_blocks = nn.ModuleList([SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples) for n_upsamples in [0, 1, 2, 3]])
        self.last_conv = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        upsample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
        return upsample(x) + y

    def forward(self, x):
        c1 = self.maxpool(self.conv_down1(x))
        c2 = self.maxpool(self.conv_down2(c1))
        c3 = self.maxpool(self.conv_down3(c2))
        c4 = self.maxpool(self.conv_down4(c3))
        c5 = self.maxpool(self.conv_down5(c4))
        p5 = self.toplayer(c5)
        p4 = self.upsample_add(p5, self.latlayer1(c4))
        p3 = self.upsample_add(p4, self.latlayer2(c3))
        p2 = self.upsample_add(p3, self.latlayer3(c2))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p2, p3, p4, p5])]
        out = self.upsample(self.last_conv(sum(feature_pyramid)), 4 * p2.size(2), 4 * p2.size(3))
        return torch.sigmoid(out)

# Example Usage
fpn = FPN().to(device)
output = fpn(torch.randn(1, 3, 256, 256).to(device))
print(output.shape)

class ResNeXtUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.last_conv0 = ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)

    def forward(self, x):
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.last_conv0(d1)
        out = self.last_conv1(out

)
        return out
```

---

### Next Steps

**a.** You could add specific unit tests for each class and function, focusing on edge cases such as different input sizes, or handling of invalid inputs.

**b.** Consider adding proper typing hints to improve code readability and error prevention.


### Code Review and Suggestions

The provided code is designed to train and evaluate deep learning models for image segmentation tasks. Here's a brief review and breakdown of its key components, and suggested improvements:

#### Key Components:
1. **Train Model**:
   - `train_model()` trains a model for segmentation, printing the training and validation metrics during each epoch.
   - The model’s performance is evaluated using a custom dice coefficient metric (likely for segmentation evaluation).
   - The learning rate is scheduled with a warm-up strategy.
   - The model weights are saved at each epoch (currently commented out).

2. **Compute IoU**:
   - The `compute_iou()` function evaluates the model’s performance on the validation dataset by calculating the average Intersection over Union (IoU) score.
   
3. **Optimizers & Learning Rate Scheduler**:
   - The `Adamax` optimizer is used for various models, and the `warmup_lr_scheduler` adjusts the learning rate during the early training steps.

4. **Plot History**:
   - The `plot_model_history()` function visualizes the training and validation DICE scores over the epochs.
   
5. **Test Prediction**:
   - After training, the models are evaluated on a test dataset, with IoU scores printed for each model.
   - Random test samples are visualized, including the predicted mask and the thresholded prediction.

---

### Suggestions & Next Steps

1. **Handle Device Compatibility**:
   - You should ensure that the device (e.g., GPU/CPU) is properly specified for both training and testing. While `data.to(device)` is used, `device` should be defined explicitly before the training loop.
     ```python
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     ```

2. **Model Saving**:
   - The model is commented out in the `train_model()` function:
     ```python
     # torch.save(model.state_dict(), f'{model_name}_{str(epoch)}_epoch.pt')
     ```
     You should uncomment this if you intend to save the model at each epoch. Ensure the file naming avoids overwriting by using the epoch as part of the filename.

3. **Edge Case Handling**:
   - The `train_model()` function could fail if `train_loader` or `val_loader` are empty or improperly formatted. It's good practice to add checks to ensure these are valid.

4. **Performance Monitoring**:
   - You can improve the performance of training by monitoring the training loss and validation loss closely. Consider implementing early stopping if the validation loss stops improving after several epochs.

5. **Plot Customization**:
   - You are using a default plot size of `(10, 6)` in the `plot_model_history()` function. You might want to pass this as an argument for flexibility.

6. **Unit Testing**:
   - Add unit tests for the helper functions (e.g., `compute_iou()`, `dice_coef_metric()`, etc.) to verify the correctness of the logic. Edge cases like empty datasets, very small inputs, etc., should be handled.

7. **GPU Memory Management**:
   - Make sure to call `torch.cuda.empty_cache()` periodically to clear up GPU memory when training on large datasets. This can prevent memory overflow during training.

8. **Threshold Value**:
   - The threshold value of `0.3` used in `compute_iou()` and for post-processing predictions might be model-specific. You may want to experiment with different thresholds and make it configurable for better control.

---

### Final Code Update

```python
# Ensure device compatibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train Model Function
def train_model(model_name, model, train_loader, val_loader, train_loss, optimizer, lr_scheduler, num_epochs):
    print(model_name)
    loss_history = []
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        model.train()  # Enter train mode
        losses = []
        train_iou = []

        if lr_scheduler:
            warmup_factor = 1.0 / 100
            warmup_iters = min(100, len(train_loader) - 1)
            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        for i_step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

            train_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
            loss = train_loss(outputs, target)

            losses.append(loss.item())
            train_iou.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

        val_mean_iou = compute_iou(model, val_loader)

        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)

        print(f"Epoch [{epoch}]")
        print(f"Mean loss on train: {np.array(losses).mean()}")
        print(f"Mean DICE on train: {np.array(train_iou).mean()}")
        print(f"Mean DICE on validation: {val_mean_iou}")

        # Save model after each epoch
        torch.save(model.state_dict(), f'{model_name}_{str(epoch)}_epoch.pt')

    return loss_history, train_history, val_history

# IoU Computation for Evaluation
def compute_iou(model, loader, threshold=0.3):
    valloss = 0
    with torch.no_grad():
        for i_step, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += picloss

    return valloss / i_step

# Training and Model History Plotting
def plot_model_history(model_name, train_history, val_history, num_epochs, figsize=(10, 6)):
    x = np.arange(num_epochs)
    fig = plt.figure(figsize=figsize)
    plt.plot(x, train_history, label='train dice', lw=3, c="springgreen")
    plt.plot(x, val_history, label='validation dice', lw=3, c="deeppink")
    plt.title(f"{model_name}", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("DICE", fontsize=15)
    plt.show()

# Additional suggestions to test performance or add optimizations
```

### Next Steps:

**a.** Add unit tests for `train_model()`, `compute_iou()`, and `dice_coef_metric()` to ensure they handle edge cases.

**b.** Implement early stopping or checkpointing to optimize training.


### Plan and Explanation of Part 5

**1. Batch Prediction and Overlap Function**  
The function `batch_preds_overlap` is used to compute the predictions and their overlaps with ground truth for a set of test samples. It returns a list of images that display the contours of both the ground truth and predicted regions:
- **Input:** Test sample images (ground truth and predicted), and model.
- **Output:** List of images with contours of both ground truth and predictions overlapped.

**2. Image Stacking and Plate Generation**  
You generate images with predictions and ground truth overlapped in grids (5x1 and 5x3). This is done for visualization purposes:
- **Output:** Stacked images for visualization, which are then saved as images.

**3. GIF Creation**  
The `make_gif` function takes images generated in the previous step and creates an animated GIF from them. It uses the `PIL` library to read and create GIFs from a sequence of images:
- **Input:** Image files.
- **Output:** GIF file.

**4. Visualization**  
Once the GIFs are created, they can be displayed using the `IPython.display` module to visualize the result in a Jupyter notebook.

---

### Full Code

```python
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import time
from IPython.display import Image as Image_display


def batch_preds_overlap(model, samples):
    """
    Computes prediction on the dataset

    Returns: list with images overlapping with predictions
    """
    prediction_overlap = []
    # model.eval()  # Optional: set model to evaluation mode
    for test_sample in samples:
        # Read image
        image = cv2.resize(cv2.imread(test_sample[1]), (128, 128))
        image = image / 255.

        # Ground truth mask
        ground_truth = cv2.resize(cv2.imread(test_sample[2], 0), (128, 128)).astype("uint8")

        # Prediction
        prediction = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
        prediction = model(prediction.to(device).float())
        prediction = prediction.detach().cpu().numpy()[0, 0, :, :]

        # Threshold prediction
        prediction[np.nonzero(prediction < 0.3)] = 0.0
        prediction[np.nonzero(prediction >= 0.3)] = 255.  # Threshold to binary
        prediction = prediction.astype("uint8")

        # Overlap with ground truth contours
        original_img = cv2.resize(cv2.imread(test_sample[1]), (128, 128))

        _, thresh_gt = cv2.threshold(ground_truth, 127, 255, 0)
        _, thresh_p = cv2.threshold(prediction, 127, 255, 0)
        contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        overlap_img = cv2.drawContours(original_img, contours_gt, 0, (0, 255, 0), 1)  # Green for GT
        overlap_img = cv2.drawContours(overlap_img, contours_p, 0, (255, 36, 0), 1)  # Red for prediction
        prediction_overlap.append(overlap_img)

    return prediction_overlap


# Create stacked images for visualization
pred_overlap_5x1_r = []
for i in range(5, 105 + 5, 5):
    pred_overlap_5x1_r.append(np.hstack(np.array(prediction_overlap_r[i - 5:i])))

pred_overlap_5x3_r = []
for i in range(3, 21 + 3, 3):
    pred_overlap_5x3_r.append(np.vstack(pred_overlap_5x1_r[i - 3:i]))

# Function to plot plate overlap
def plot_plate_overlap(batch_preds, title, num):
    plt.figure(figsize=(15, 15))
    plt.imshow(batch_preds)
    plt.axis("off")

    plt.figtext(0.76, 0.75, "Green - Ground Truth", va="center", ha="center", size=20, color="lime")
    plt.figtext(0.26, 0.75, "Red - Prediction", va="center", ha="center", size=20, color="#ff0d00")
    plt.suptitle(title, y=.80, fontsize=20, weight="bold", color="#00FFDE")

    fn = "_".join((title + str(num)).lower().split()) + ".png"
    plt.savefig(fn, bbox_inches='tight', pad_inches=0.2, transparent=False, facecolor='black')
    plt.close()


# Plotting the result for ResNeXt50
title3 = "Predictions of UNet with ResNeXt50 backbone"
for num, batch in enumerate(pred_overlap_5x3_r):
    plot_plate_overlap(batch, title3, num)


# Create a GIF from the predictions (ResNeXt50 in this case)
def make_gif(title):
    base_name = "_".join(title.lower().split())

    base_len = len(base_name)
    end_len = len(".png")
    fp_in = f"{base_name}*.png"
    fp_out = f"{base_name}.gif"

    img, *imgs = [Image.open(f)
                  for f in sorted(glob.glob(fp_in), key=lambda x: int(x[base_len:-end_len]))]

    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=1000, loop=0)

    return fp_out


# Create gif for ResNeXt50
fn3 = make_gif(title3)

# Display gif
with open(fn3, 'rb') as f:
    display(Image_display(data=f.read(), format='png'))
```

### Key Points in This Code:
1. **Model Inference and Overlap:**  
   - The `batch_preds_overlap` function handles the prediction of images from the test set and computes the overlap between the predicted and ground truth contours. Contours are drawn for both the predicted and ground truth images to visualize the overlap.
   
2. **Image Grid Creation:**  
   - The images are stacked into 5x1 and 5x3 grids for better visualization. This helps in comparing predictions across multiple test images.

3. **GIF Generation:**  
   - The `make_gif` function collects the images in order, creates a GIF, and allows you to visualize the progression of predictions over time in an animated format.

4. **Visualization of Results:**  
   - The `plot_plate_overlap` function creates the visualization and saves the images. It also adds labels for the ground truth (green) and predicted (red) regions.

---

### Suggestions for Next Steps:
**a.** Explore experimenting with different thresholds (other than 0.3) to see how that affects prediction accuracy and visualization.  
**b.** Test the performance of your model by increasing the batch size or image size to see if it influences the results positively.
