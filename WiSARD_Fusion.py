import torch
from torchvision import transforms
from PIL import Image
from FusionNet import FusionNet
from Gen_attn import Gen
import os
from Metrics import SSIM
import matplotlib.pyplot as plt

# Define the path to the saved model
# model_path = "SeAFusion/model/Fusion/fusionmodel_final.pth"
model_path = "SeAFusion/gen_16.pth.tar"

# Define the paths to the input folders containing visual and thermal images
visual_folder = "D:/Your Masters/Summer 22/GAN/Data/Vis/train"
thermal_folder = "D:/Your Masters/Summer 22/GAN/Data/ir/train"

# Define the output folder to save the fused images
output_folder = "Aad/fused"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the transformation to apply to the input images
transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor()
])

# Load the saved model
# model = FusionNet(output=1)
model = Gen()
# model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"])
model.eval()

# Get the list of image files in the visual folder (assuming both folders have the same number of images)
visual_files = sorted(os.listdir(visual_folder))
thermal_files = sorted(os.listdir(thermal_folder))

# Iterate over the images
for visual_file, thermal_file in zip(visual_files, thermal_files):
    # Construct the paths to the visual and thermal images
    visual_image_path = os.path.join(visual_folder, visual_file)
    thermal_image_path = os.path.join(thermal_folder, thermal_file)

    # Load and preprocess the input images
    input_image1 = transform(Image.open(visual_image_path)).unsqueeze(0)
    input_image2 = transform(Image.open(thermal_image_path)).unsqueeze(0)

    ssim_ir = []
    ssim_vis = []

    # Perform image fusion
    with torch.no_grad():
        fused_image,_,_,_ = model(input_image1, input_image2)
        ssim_vis.append(SSIM(fused_image, input_image1))
        ssim_ir.append(SSIM(fused_image, input_image2))


    # Save the fused image
    fused_image = fused_image.squeeze().detach().cpu()
    output_image_path = os.path.join(output_folder, visual_file)
    transforms.ToPILImage()(fused_image).save(output_image_path)

    print("Image fusion complete. Fused image saved at:", output_image_path)

## Create a figure and axes
# ssim_ir = [0.85, 0.92, 0.78, 0.81, 0.89, 0.95, 0.88, 0.44, 0.65, 0.75, 0.45, 0.45, 0.45]
# fig, ax = plt.subplots()

# # Plot box plots
# ax.boxplot([ssim_ir], vert = False, labels=['IR Images'])

# # Set y-axis label
# ax.set_ylabel('SSIM Values')

# # Set title
# ax.set_title('Distribution of SSIM Values')

# # Show the plot
# plt.show()
