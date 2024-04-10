import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from models import ViT

def visualize_results(model_path, testloader, classes, num_images=36):
    # Load the trained model
    net = ViT()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # Get a batch of test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Make predictions on the test images
    images = images.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Create a montage of the test images with labels
    montage = make_grid(images[:num_images], nrow=6, padding=2).cpu()
    montage_image = transforms.ToPILImage()(montage)

    # Add labels to the montage
    draw = ImageDraw.Draw(montage_image)
    font = ImageFont.truetype("arial.ttf", 12)

    for i in range(num_images):
        x = i % 6 * montage_image.width // 6 + 5
        y = i // 6 * montage_image.height // 6 + 5
        label_text = f'Truth: {classes[labels[i]]}\nPredicted: {classes[predicted[i]]}'
        draw.text((x, y), label_text, font=font, fill='black')

    # Save the montage as "result.png"
    result_path = os.path.join(os.path.dirname(model_path), 'result.png')
    montage_image.save(result_path)
    print(f'Montage of test images with labels saved to {result_path}')