import os

# Path to the directories for the three different compression levels
compression_100_path_train = './jpeg-compressions/level-100/test'
compression_50_path_train = './jpeg-compressions/level-50/test'
compression_10_path_train = './jpeg-compressions/level-10/test'

# Path to directory for the original images
original_images_path = './tiny-imagenet/test/images'

# List of all images in the Tiny-Imagenet test dataset
images = os.listdir(compression_100_path_train)

compression_ratios_100 = []
compression_ratios_50 = []
compression_ratios_10 = []
# Loop through all images in the test dataset, get AVERAGE compression ratios
for image_name in images:
    # Path to the original image
    original_image_path = os.path.join(original_images_path, image_name)
    original_image_size = os.path.getsize(original_image_path)

    # Path to the compressed images
    compressed_image_100_path = os.path.join(compression_100_path_train, image_name)
    compressed_image_50_path = os.path.join(compression_50_path_train, image_name)
    compressed_image_10_path = os.path.join(compression_10_path_train, image_name)

    # Get the sizes of the compressed images
    compressed_image_100_size = os.path.getsize(compressed_image_100_path)
    compressed_image_50_size = os.path.getsize(compressed_image_50_path)
    compressed_image_10_size = os.path.getsize(compressed_image_10_path)

    # Calculate the compression ratios
    compression_ratio_100 = original_image_size / compressed_image_100_size
    compression_ratio_50 = original_image_size / compressed_image_50_size
    compression_ratio_10 = original_image_size / compressed_image_10_size

    # Append the compression ratios to the lists
    compression_ratios_100.append(compression_ratio_100)
    compression_ratios_50.append(compression_ratio_50)
    compression_ratios_10.append(compression_ratio_10)

# Calculate the AVERAGE compression ratios
average_compression_ratio_100 = sum(compression_ratios_100) / len(compression_ratios_100)
average_compression_ratio_50 = sum(compression_ratios_50) / len(compression_ratios_50)
average_compression_ratio_10 = sum(compression_ratios_10) / len(compression_ratios_10)

# Print the AVERAGE compression ratios
print('Average compression ratio at 70%:', average_compression_ratio_100)
print('Average compression ratio at 30%:', average_compression_ratio_50)
print('Average compression ratio at 10%:', average_compression_ratio_10)