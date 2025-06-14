# Galaxy Classification CNN

A Convolutional Neural Network (CNN) built in PyTorch to classify astronomical images into 3 classes:

- Elliptical Galaxy
- No Galaxy / Stars
- Spiral Galaxy

The project includes full data preprocessing, augmentation, training, evaluation, and visualization of some learned feature maps.
More work needs to be done to the data that I am using. Since some images contain both types of galaxy classes it can cause some issues.
Another issue with the data is that I obtained most of it by picking small areas in the sdss skyserver and slightly adjusting the degree at which images that met the criteria was met, meaning that a galaxy that was center of one image might be in the next image slightly off center as the "scan" continued.

---
## Positive Class (Galaxies) Data Collection

The positive samples for the model — images containing galaxies — were primarily obtained from the Galaxy Zoo 1 dataset. 
Galaxy Zoo is a citizen science project where volunteers visually classify galaxies from the Sloan Digital Sky Survey (SDSS). 
Because multiple volunteers review each image, Galaxy Zoo provides probabilistic classifications indicating how confident the community is that a given object belongs to a particular morphological class.

To ensure high-quality and reliable labels for training, I applied a probability threshold of 0.85. 
This means I selected only those galaxies for which at least 85% of the classifiers agreed on the object's classification. 
This helped reduce label noise and improved the likelihood that the images used for training were correctly categorized.

I retrieved the corresponding SDSS images using the SDSS image cutout service.
- **Image size**: 256x256 pixels
- **Pixel scale**: 0.396 arcseconds/pixel (which is the native SDSS resolution)

By keeping the native pixel scale, I preserved the physical size of the galaxies on the sky, ensuring that spatial information remained consistent across samples.

## Negative Class (No Galaxies) Data Collection and its issues.

One of the challenges in training my model was obtaining a sufficient quantity of negative examples — that is, images that do not contain galaxies. 
Since most astronomical catalogs (like SDSS) are focused on detecting and cataloging objects, obtaining true negatives (empty sky or star-only fields) requires some careful filtering.

To create this negative dataset, I queried the SDSS database using SciServer/CASJobs.
The SQL query I used performs a LEFT JOIN between the SDSS Field table (which contains information about image tiles) and the PhotoObjAll table (which lists all detected astronomical objects, including galaxies). 
The goal was to find fields where no galaxies were detected.
The key part of the query logic is:

- For each field (Field table), I checked whether any galaxies (PhotoObjAll where type=3) fell within that field's RA/DEC boundaries.
- If no galaxy was present (p.objID IS NULL after the LEFT JOIN), I selected that field as a candidate negative example.
- To ensure better image quality, I filtered for fields where f.score > 0.2 to avoid poor seeing conditions or artifacts.
- I limited the total number of rows returned to prevent excessively large datasets during initial runs

### The Subtle Issue With This Approach
Since the SDSS field tiling overlaps slightly as it scans the sky, some galaxies that are centered in one field may appear near the edges of neighboring fields. 
This means that even though my query correctly identified fields without a galaxy detection in the center, a galaxy may still exist partially off-center in the image due to overlapping scan boundaries.
In effect, some images classified as "no galaxies" may still contain galaxies near the edge of the frame, or galaxies that simply failed SDSS's detection threshold in that particular field.

This is an inherent limitation of using field-based selection for negative sampling:

- The absence of a galaxy detection (p.objID IS NULL) does not always guarantee that no galaxy is visually present in the field.
- Galaxies near the detection limit, partially resolved galaxies, or objects not classified as type=3 may still be present.
- Overlapping scan boundaries (due to how SDSS surveys the sky) can introduce positional drift between adjacent fields.

Despite these challenges, this method still provides a reasonably large set of likely negative examples suitable for model training, especially when combined with data augmentation and additional filtering steps during preprocessing.
I also did my best to manually remove images that obviously contained galaxies.





## Model Architecture

The CNN architecture consists of multiple convolutional layers followed by global pooling and a fully connected classifier.

### Convolutional Layers

| Layer        | Input Channels | Output Channels | Kernel Size | Stride | Padding |
| ------------ | -------------- | --------------- | ----------- | ------ | ------- |
| `conv1`      | 3              | 40              | 3           | 1      | 1       |
| `conv1_down` | 40             | 64              | 3           | 1      | 1       |
| `conv2_down` | 64             | 96              | 3           | 2      | 1       |
| `conv3_down` | 96             | 160             | 3           | 2      | 1       |
| `conv4_down` | 160            | 256             | 3           | 2      | 1       |

- Each layer uses Batch Normalization and ReLU activations.
- Downsampling is achieved via strided convolutions.
- Adaptive Average Pooling reduces the output to a fixed 256-dim feature vector.

### Fully Connected Layer

- `fc1`: 256 -> 3 (output classes)

---

## Data Pipeline

### Image Size

- All images are resized to **128x128** for ease of training but can be adjusted for better or worse hardware.

### Data Augmentation (Training)

- Random rotations (0-360 degrees)
- Random translations (up to 20% shift in both axes)

### Validation Data

- Only resized, no augmentation.

---

## Dataset Balancing

The dataset is highly imbalanced:

| Class      | Image Count |
| ---------- | ----------- |
| Elliptical | 23,335      |
| No Galaxy  | 15,472      |
| Spiral     | 53,022      |

To counter imbalance, a **WeightedRandomSampler** oversamples underrepresented classes during training.

---

## Training Configuration

| Parameter       | Value                      |
| --------------- | -------------------------- |
| Epochs          | 6                          |
| Batch Size      | 240                        |
| Optimizer       | Adam                       |
| Learning Rate   | 0.001                      |
| Weight Decay    | 1e-4                       |
| Scheduler       | StepLR (step=3, gamma=0.7) |
| Mixed Precision | PyTorch AMP                |

---

## Hardware

- Fully utilizes CUDA (GPU) if available.
- Optimized and tested on NVIDIA RTX 5090.

---

## Activation Map Visualization

After training, feature maps from each convolutional layer can be visualized for any input image. This helps to inspect how the network detects edges, textures, shapes, and structures at different abstraction levels.

---

## Output

- Model weights are saved to: `galaxy_cnn_225.pth`
- Test accuracy is printed after training.

---

## Dependencies

- `torch`
- `torchvision`
- `matplotlib`
- `os`

---

## Running the Model

```bash
python CNN-Galaxy.py
```

This will:

- Load data
- Train the CNN
- Evaluate model accuracy
- Save the trained weights
- Generate activation map visualizations

---

## Notes

- Dataset paths are set via `PATH_TO_DATA_TRAIN` and `PATH_TO_DATA_VALIDATION`.
- Dataset must follow the folder structure used by `torchvision.datasets.ImageFolder`.
- Code can easily be modified to adjust architecture, hyperparameters, and dataset locations.

