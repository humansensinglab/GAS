import torchvision.transforms as transforms
from PIL import Image, ImageOps
import sys

def pad_to_square(image_path, output_path, fill_color=(0, 0, 0)):
    """
    Pads a PNG image to make it square while keeping it centered.

    :param image_path: Path to the input PNG image.
    :param output_path: Path to save the output squared image.
    :param fill_color: Background color for padding (default is black).
    """
    img = Image.open(image_path)

    # Get dimensions
    width, height = img.size
    max_side = max(width, height)

    # Calculate padding
    pad_left = (max_side - width) // 2
    pad_top = (max_side - height) // 2
    pad_right = max_side - width - pad_left
    pad_bottom = max_side - height - pad_top

    # Add padding
    squared_img = ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill_color)

    # Save the output image
    squared_img.save(output_path, format="PNG")

if __name__ == "__main__":
    pad_to_square(sys.argv[1], sys.argv[1], fill_color=(255, 255, 255))

    transform = transforms.RandomResizedCrop(size=512, scale=(1.0, 1.0), ratio=(1.0, 1.0))
    image_path = sys.argv[1]
    save_path = image_path
    image = Image.open(image_path).convert("RGB")
    transformed_image = transform(image)
    transformed_image.save(save_path)