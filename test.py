import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet
import os
import numpy as np
from scipy.ndimage import gaussian_filter

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
    return output.squeeze().cpu().numpy()

def soften_edges(mask, sigma=1):
    return gaussian_filter(mask, sigma=sigma)

def apply_mask(image, mask):
    # Create an RGBA image
    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
    rgba[:,:,:3] = image
    rgba[:,:,3] = mask  # Use the mask values directly as alpha
    
    return rgba

def save_output(original, mask, softened_mask, result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image
    Image.fromarray((original * 255).astype(np.uint8)).save(os.path.join(output_dir, 'original.png'))
    
    # Save original mask
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(output_dir, 'mask_original.png'))
    
    # Save softened mask
    Image.fromarray((softened_mask * 255).astype(np.uint8)).save(os.path.join(output_dir, 'mask_softened.png'))
    
    # Save result (background removed with transparency)
    Image.fromarray((result * 255).astype(np.uint8)).save(os.path.join(output_dir, 'result_transparent.png'))
    
    print(f"Results saved in '{output_dir}' directory")

def main():
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load('best_background_removal_model.pth'))
    
    # Load and process the test image
    image_path = 'D:/AIEMB/in/360_F_266724172_Iy8gdKgMa7XmrhYYxLCxyhx6J7070Pr8.jpg'
    image = load_image(image_path).to(device)
    
    # Predict mask
    mask = predict(model, image)
    
    # Soften the edges of the mask
    softened_mask = soften_edges(mask, sigma=1)
    
    # Apply softened mask to original image
    original = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    result = apply_mask(original, softened_mask)
    
    # Visualize results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 7))
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Original Mask')
    ax3.imshow(softened_mask, cmap='gray')
    ax3.set_title('Softened Mask')
    ax4.imshow(result)
    ax4.set_title('Image with Transparent Background')
    
    for ax in (ax1, ax2, ax3, ax4):
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save output to files
    output_dir = 'D:/AIEMB/out'  # You can change this to your desired output directory
    save_output(original, mask, softened_mask, result, output_dir)

if __name__ == '__main__':
    main()