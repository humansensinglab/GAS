import torchvision.transforms as transforms
from PIL import Image, ImageOps
import sys
from pathlib import Path

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

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in ['.png']
if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    output_dir = input_path if input_path.is_dir() else input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_to_process = []

    if input_path.is_file():
        if is_image_file(input_path):
            paths_to_process = [input_path]
        else:
            raise ValueError(f"Not an image: {input_path}")
    elif input_path.is_dir():
        for p in input_path.iterdir():
            if p.is_file() and is_image_file(p):
                paths_to_process.append(p)
        if not paths_to_process:
            print(f"No image file inside this dir")
    else:
        raise ValueError("Invalid input path")
    
    transform = transforms.Resize((512, 512), antialias=True)
    for img_path in paths_to_process:
        pad_to_square(img_path, img_path, fill_color=(255, 255, 255))
        image = Image.open(img_path).convert("RGB")
        transformed_image = transform(image)
        transformed_image.save(img_path)