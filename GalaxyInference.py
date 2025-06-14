import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

root = tk.Tk()
root.withdraw()  # Hide root window
file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])


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
            self.fc1 = nn.Linear(256, 3)
            self.activations = {}
            
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
        
# Recreate model and load weights
model = CNN()
model.load_state_dict(torch.load('galaxy_cnn.pth'))
model.eval()

img1 = Image.open(file_path).convert('RGB')
img = transform(img1)
img = img.unsqueeze(0)

def unnormalize(img, mean, std):
        # img shape: [C, H, W], values in normalized space
        img = img.clone()  # avoid modifying the original
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        return img



with torch.no_grad():
    outputs = model(img, visualize=True)
    _, predicted = torch.max(outputs, 1)

# Class mapping (same order as your training dataset)
classes = ['Elliptical', 'No Galaxy', 'Spiral']
predicted_class = classes[predicted.item()]

print("Predicted class:", predicted_class)
predicted_name = predicted_class



subplots = []
def on_click(event):
    for ax, activation_img in subplots:
        if event.inaxes == ax:
            plt.figure(figsize=(6,6))
            plt.imshow(activation_img, cmap='viridis')
            plt.title("Zoomed Activation")
            plt.axis('off')
            plt.show()
            break


def plot_combined_feature_maps(activations, input_image, layers=('conv1', 'conv2', 'conv3', 'conv4', 'conv5'), num_maps=8, predicted_label=None):
    total_layers = len(layers)
    plt.figure(figsize=((num_maps + 2) * 2, total_layers * 2.5))

    plt.subplot(total_layers, num_maps + 1, 1)
    plt.imshow(input_image.permute(1, 2, 0).numpy())
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')

    for row, layer_name in enumerate(layers):
        feature_map = activations[layer_name][0]
        channels = feature_map.shape[0]
        for col in range(min(num_maps, channels)):
            plt_idx = row * (num_maps + 1) + col + 2
            ax = plt.subplot(total_layers, num_maps + 1, plt_idx)
            activation_img = feature_map[col].cpu().numpy()
            ax.imshow(activation_img, cmap='viridis')
            ax.axis('off')
            if row == 0:
                plt.title(f'Kernel {col}')

            # Store each subplot and image for click handling
            subplots.append((ax, activation_img))

    plt.suptitle("CNN Feature Maps with Input", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Connect mouse click
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()

img_for_plot = img.squeeze(0).cpu()
plot_combined_feature_maps(model.activations, img_for_plot, predicted_label=predicted_name)