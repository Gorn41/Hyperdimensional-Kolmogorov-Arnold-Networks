import os
from glob import glob
from PIL import Image
from tqdm import tqdm

# Path to the directory containing the ImageNet-64 images
imagenet64_dir = 'tiny-imagenet'

# Paths to the directories for the two compression levels
compression_60 = 'webp-compressions/compression_60'
compression_30 = 'webp-compressions/compression_30'

# Ensure output directories exist
os.makedirs(compression_60, exist_ok=True)
os.makedirs(compression_30, exist_ok=True)

# Recursively find all images (JPEG) in the directory and its subdirectories
image_paths = glob(os.path.join(imagenet64_dir, '**', '*.JPEG'), recursive=True)

print(f"Found {len(image_paths)} images")

# Loop through all images in the directory
for img_path in tqdm(image_paths, desc='Converting to WebP at quality levels 60 and 30'):
    # Determine relative path of image within the original directory structure
    relative_path = os.path.relpath(img_path, imagenet64_dir)
    
    # Replace the extension with .webp
    filename_60 = os.path.splitext(relative_path)[0] + '.webp'
    filename_30 = os.path.splitext(relative_path)[0] + '.webp'

    # Determine the full output paths for each quality setting
    output_path_60 = os.path.join(compression_60, filename_60)
    output_path_30 = os.path.join(compression_30, filename_30)

    # Create any necessary subdirectories in the output path
    os.makedirs(os.path.dirname(output_path_60), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_30), exist_ok=True)

    try:
        # Load image and convert to RGB
        img = Image.open(img_path).convert('RGB')
        
        # Save image at quality level 60
        img.save(output_path_60, 'webp', quality=60)
        
        # Save image at quality level 30
        img.save(output_path_30, 'webp', quality=30)

    except Exception as e:
        print(f"Error converting {img_path}: {e}")
