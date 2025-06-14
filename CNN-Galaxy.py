import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import os
import matplotlib.pyplot as plt
from torch import amp

IMAGE_RESIZE = 128   # My RTX 5090 does fine with this value, a better setup can handle higher values.
PATH_TO_DATA_TRAIN = 'dataset/train'
PATH_TO_DATA_VALIDATION = 'dataset/val'


# The count for the unbalanced training dataset. [eliptical, no galaxy/stars, spiral]
# Gets used with the Weighted Random Sampler 
# I might reduce Spiral galaxy class in the future
CLASS_COUNT = [23335, 15472, 53022]


# --Training Parameters--
NUM_EPOCHS = 6   # low numbers seem here to works fine but indicates some issues with the data.
LEARNING_RATE = 0.001   # LR for Adam optimizer
WEIGHT_DECAY = 1e-4
# for scheduler lr
SCHEDULE_STEP = 3
GAMMA = 0.7







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == 'cuda':
    print(" -> Name:", torch.cuda.get_device_name(0))
    print(" -> Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**2, 1), "MB")
    print(" -> Memory Cached:   ", round(torch.cuda.memory_reserved(0)/1024**2, 1), "MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Safer folder (you can change this to something simple like "C:/CNN/data")
root_dir = os.path.abspath("data")
os.makedirs(root_dir, exist_ok=True)




def train():


    # # --- Define Transforms ---

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        transforms.RandomRotation(degrees=360),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
    ])


    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(PATH_TO_DATA_TRAIN, transform=train_transform)
    test_dataset = datasets.ImageFolder(PATH_TO_DATA_VALIDATION, transform=val_transform)

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=40, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(40)
            self.relu1 = nn.ReLU()

            self.conv1_down = nn.Conv2d(40, 64, 3, padding=1, stride=1)   # Change out_channels if you want more features
            self.bn1_down = nn.BatchNorm2d(64)
            self.relu1_down = nn.ReLU() # 16x16

            self.conv2_down = nn.Conv2d(64, 96, 3, padding=1, stride=2) 
            self.bn2_down = nn.BatchNorm2d(96)
            self.relu2_down = nn.ReLU() # 8x8

            self.conv3_down = nn.Conv2d(96, 160, 3, padding=1, stride=2) 
            self.bn3_down = nn.BatchNorm2d(160)
            self.relu3_down = nn.ReLU() # 4x4

            self.conv4_down = nn.Conv2d(160, 256, 3, padding=1, stride=2) 
            self.bn4_down = nn.BatchNorm2d(256)
            self.relu4_down = nn.ReLU() # 2x2

            self.global_pool = nn.AdaptiveAvgPool2d(1)


            # Fully connected layers
            self.fc1 = nn.Linear(256, 3)
            
        def forward(self, input, visualize=False):
            if visualize:
                self.activations = {}


            x = self.relu1(self.bn1(self.conv1(input)))
            

            if visualize:
                self.activations['conv1'] = x.detach().cpu()

            x = self.relu1_down(self.bn1_down(self.conv1_down(x)))
            if visualize:
                self.activations['conv2'] = x.detach().cpu()

            x = self.relu2_down(self.bn2_down(self.conv2_down(x)))
            if visualize:
                self.activations['conv3'] = x.detach().cpu()
            
            x = self.relu3_down(self.bn3_down(self.conv3_down(x)))
            if visualize:
                self.activations['conv4'] = x.detach().cpu()

            x = self.relu4_down(self.bn4_down(self.conv4_down(x)))
            if visualize:
                self.activations['conv5'] = x.detach().cpu()

            
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)  # Flatten
        
            x = self.fc1(x)
            
        


            return x
        


    # The counts for the unbalanced dataset. [eliptical, no galaxy/stars, spiral]
    class_counts = CLASS_COUNT
    total_samples = sum(class_counts)
    class_weights = [total_samples / c for c in class_counts]

    # Map weights to each sample
    targets = train_dataset.targets  # This comes directly from ImageFolder
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=240, sampler=sampler)

    test_loader = DataLoader(test_dataset, batch_size=1000)


    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULE_STEP, gamma=GAMMA)
    scaler = amp.GradScaler("cuda")

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                  
        scheduler.step()

        print(f"Epoch [{epoch+1}/{str(NUM_EPOCHS)}], Loss: {loss.item():.4f}")



    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Save the model weights
    torch.save(model.state_dict(), 'galaxy_cnn_225.pth')
    print('Model saved')


    image, labels = next(iter(train_loader))
    image = image[0].unsqueeze(0).to(device)  # Get just 1 image, shape [1, 1, 28, 28]
    label = labels[0].item() 
    model(image, visualize=True)  # Forward pass with introspection
    output = model(image)
    GAL_CLASSES = [
        'Elliptical', 'No Galaxy', 'Spiral'
    ]
    true_name = GAL_CLASSES[label]
    predicted_index = output.argmax(dim=1).item()
    predicted_name = GAL_CLASSES[predicted_index]

    print("Activations keys:", model.activations.keys())
    def plot_combined_feature_maps(activations, input_image, layers=('conv1', 'conv2', 'conv3', 'conv4', 'conv5'), num_maps=8, predicted_label=None):
        total_layers = len(layers)
        plt.figure(figsize=((num_maps + 2) * 2, total_layers * 2.5))

        # Plot the original image in the top-left
        input_img = input_image.permute(1, 2, 0).numpy()   # This is now a numpy array
        plt.subplot(total_layers, num_maps + 1, 1)
        plt.imshow(input_img)

        plt.title(f"Input: {true_name}\n Prediction: {predicted_name}")
        plt.axis('off')

        # Now plot filters starting at col=2
        for row, layer_name in enumerate(layers):
            feature_map = activations[layer_name][0]  # [C, H, W]
            channels = feature_map.shape[0]
            for col in range(min(num_maps, channels)):
                plt_idx = row * (num_maps + 1) + col + 2  # +2 to account for 1-based index and input image
                plt.subplot(total_layers, num_maps + 1, plt_idx)
                plt.imshow(feature_map[col], cmap='viridis')
                plt.axis('off')
                if row == 0:
                    plt.title(f'Kernel {col}')
            # Label layer name vertically on the left (optional)
            if num_maps > 0:
                plt.text(-0.1, 0.5 + row * (num_maps + 1), layers[row], va='center', ha='right',
                        fontsize=10, transform=plt.gcf().transFigure)
                

        plt.suptitle("CNN Feature Maps with Input", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()


    img_for_plot = image.squeeze(0).cpu()

    plot_combined_feature_maps(model.activations, img_for_plot, predicted_label=predicted_name)


if __name__ == '__main__':
    train()