# Compresses all images in a directory into JPEG-2000s at three different compression levels: 1000, 500, and 100.
import os
import cv2
import shutil

mid_quality_level = 30
low_quality_level = 13

# TEST FILES

# Path to the directory containing the Tiny-Imagenet test images
tiny_imagenet_path_test = './tiny-imagenet/test/images'

# Path to the directories for the three different compression levels
compression_mid_quality_path_test = './JPEG-compressions/mid_quality/test/images'
compression_low_quality_path_test = './JPEG-compressions/low_quality/test/images'

# List of all images in the Tiny-Imagenet test dataset
images = os.listdir(tiny_imagenet_path_test)

# Loop through all images in the test dataset
for image_name in images:
    # Path to the image
    image_path = os.path.join(tiny_imagenet_path_test, image_name)

    # Read the image
    image = cv2.imread(image_path)

    # Compress the image at three different compression levels
    cv2.imwrite(os.path.join(compression_mid_quality_path_test, image_name), image, [cv2.IMWRITE_JPEG_QUALITY, mid_quality_level])
    cv2.imwrite(os.path.join(compression_low_quality_path_test, image_name), image, [cv2.IMWRITE_JPEG_QUALITY, low_quality_level])

print('All test images compressed successfully!')

# VALIDATION FILES
tiny_imagenet_path_val = './tiny-imagenet/val/images'

compression_mid_quality_path_val = './JPEG-compressions/mid_quality/val/images'
compression_low_quality_path_val = './JPEG-compressions/low_quality/val/images'

# List of all images in the Tiny-Imagenet validation dataset
images = os.listdir(tiny_imagenet_path_val)

# Loop through all images in the validation dataset
for image_name in images:
    # Path to the image
    image_path = os.path.join(tiny_imagenet_path_val, image_name)

    # Read the image
    image = cv2.imread(image_path)

    # Compress the image at three different compression levels
    cv2.imwrite(os.path.join(compression_mid_quality_path_val, image_name), image, [cv2.IMWRITE_JPEG_QUALITY, mid_quality_level])
    cv2.imwrite(os.path.join(compression_low_quality_path_val, image_name), image, [cv2.IMWRITE_JPEG_QUALITY, low_quality_level])

print('All validation images compressed successfully!')

# TRAIN FILES

# Path to the directory containing the Tiny-Imagenet images
tiny_imagenet_path_train = './tiny-imagenet/train'

# Path to the directories for the two compression levels
compression_mid_quality_path_train = './JPEG-compressions/mid_quality/train'
compression_low_quality_path_train = './JPEG-compressions/low_quality/train'

# List of all directories in the Tiny-Imagenet training dataset
directories = os.listdir(tiny_imagenet_path_train)

# Loop through all directories in the training dataset
for directory in directories:

    # Skip directories that start with a period
    if directory.startswith('.'):
        continue

    # Create a new directory for the class
    if not os.path.exists(os.path.join(compression_mid_quality_path_train, directory)):
        os.mkdir(os.path.join(compression_mid_quality_path_train, directory))
    if not os.path.exists(os.path.join(compression_low_quality_path_train, directory)):
        os.mkdir(os.path.join(compression_low_quality_path_train, directory))

    # Path to the boxes.txt file
    boxes_path = os.path.join(tiny_imagenet_path_train, directory, directory + '_boxes.txt')

    # Copy the boxes.txt file to the new directory
    if os.path.exists(boxes_path):
        shutil.copy(boxes_path, os.path.join(compression_mid_quality_path_train, directory))
        shutil.copy(boxes_path, os.path.join(compression_low_quality_path_train, directory))

    # Path to the directory containing the images
    directory_path = os.path.join(tiny_imagenet_path_train, directory, 'images')

    # Create a new directory for the images
    if not os.path.exists(os.path.join(compression_mid_quality_path_train, directory, 'images')):
        os.mkdir(os.path.join(compression_mid_quality_path_train, directory, 'images'))
    if not os.path.exists(os.path.join(compression_low_quality_path_train, directory, 'images')):
        os.mkdir(os.path.join(compression_low_quality_path_train, directory, 'images'))

    # List of all images in the directory
    images = os.listdir(directory_path)

    # Loop through all images in the directory
    for image_name in images:
        # Path to the image
        image_path = os.path.join(directory_path, image_name)

        # Read the image
        image = cv2.imread(image_path)

        # Compress the image at three different compression levels
        cv2.imwrite(os.path.join(compression_mid_quality_path_train, directory, 'images', image_name), image, [cv2.IMWRITE_JPEG_QUALITY, mid_quality_level])
        cv2.imwrite(os.path.join(compression_low_quality_path_train, directory, 'images', image_name), image, [cv2.IMWRITE_JPEG_QUALITY, low_quality_level])

print('All training images compressed successfully!')