import os

# Path to the directories for the three different compression levels
compression_mid_quality_path_train = './JPEG-compressions/mid_quality/train'
compression_low_quality_path_train = './JPEG-compressions/low_quality/train'

# Path to the directory containing the Tiny-Imagenet train folders
tiny_imagenet_path_train = './tiny-imagenet/train'

# List of all directories in the Tiny-Imagenet training dataset
directories = os.listdir(tiny_imagenet_path_train)

mid_quality_image_ratios = []
low_quality_image_rations = []

# Loop through all directories in the training dataset
for directory in directories:

    if directory.startswith('.'):
        continue

    # Path to the directory containing the images
    directory_path = os.path.join(tiny_imagenet_path_train, directory, 'images')
    
    # List of all images in the directory
    images = os.listdir(directory_path)
    
    # Loop through all images in the directory
    for image_name in images:
        # Path to the image
        image_path = os.path.join(directory_path, image_name)
        original_image_size = os.path.getsize(image_path)

        # Path to the compressed images
        mid_quality_image_path = os.path.join(compression_mid_quality_path_train, directory, 'images', image_name)
        low_quality_image_path = os.path.join(compression_low_quality_path_train, directory, 'images', image_name)

        # Calculate the compression ratio for the mid-quality compression
        mid_quality_image_size = os.path.getsize(mid_quality_image_path)
        mid_quality_image_ratios.append(mid_quality_image_size / original_image_size)

        # Calculate the compression ratio for the low-quality compression
        low_quality_image_size = os.path.getsize(low_quality_image_path)
        low_quality_image_rations.append(low_quality_image_size / original_image_size)

# Calculate the average compression ratio for the mid-quality compression
average_mid_quality_image_ratio = sum(mid_quality_image_ratios) / len(mid_quality_image_ratios)

# Calculate the average compression ratio for the low-quality compression
average_low_quality_image_ratio = sum(low_quality_image_rations) / len(low_quality_image_rations)

print('Average compression ratio for mid-quality compression on train set:', average_mid_quality_image_ratio)
print('Average compression ratio for low-quality compression on train set:', average_low_quality_image_ratio)

# Do the same for test and validation datasets 

# Path to the directories for the two different compression levels
compression_mid_quality_path_test = './JPEG-compressions/mid_quality/test/images'
compression_low_quality_path_test = './JPEG-compressions/low_quality/test/images'

# Path to the directory containing the Tiny-Imagenet test images
tiny_imagenet_path_test = './tiny-imagenet/test/images'

# List of all images in the Tiny-Imagenet test dataset
images = os.listdir(tiny_imagenet_path_test)

mid_quality_image_ratios = []
low_quality_image_rations = []

# Loop through all images in the test dataset
for image_name in images:
    # Path to the image
    image_path = os.path.join(tiny_imagenet_path_test, image_name)
    original_image_size = os.path.getsize(image_path)

    # Path to the compressed images
    mid_quality_image_path = os.path.join(compression_mid_quality_path_test, image_name)
    low_quality_image_path = os.path.join(compression_low_quality_path_test, image_name)

    # Calculate the compression ratio for the mid-quality compression
    mid_quality_image_size = os.path.getsize(mid_quality_image_path)
    mid_quality_image_ratios.append(mid_quality_image_size / original_image_size)

    # Calculate the compression ratio for the low-quality compression
    low_quality_image_size = os.path.getsize(low_quality_image_path)
    low_quality_image_rations.append(low_quality_image_size / original_image_size)

# Calculate the average compression ratio for the mid-quality compression
average_mid_quality_image_ratio = sum(mid_quality_image_ratios) / len(mid_quality_image_ratios)

# Calculate the average compression ratio for the low-quality compression
average_low_quality_image_ratio = sum(low_quality_image_rations) / len(low_quality_image_rations)

print('Average compression ratio for mid-quality compression on test set:', average_mid_quality_image_ratio)
print('Average compression ratio for low-quality compression on test set:', average_low_quality_image_ratio)