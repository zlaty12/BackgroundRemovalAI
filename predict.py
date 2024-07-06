import torch
from torchvision import transforms
from PIL import Image
from model import UNet

def predict(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    output_image = transforms.ToPILImage()(output.squeeze())
    return output_image

# Load the trained model
model = UNet()
model.load_state_dict(torch.load('background_removal_model.pth'))
model.eval()

# Predict on a new image
result = predict(model, 'path/to/test/image.jpg')
result.save('output.png')