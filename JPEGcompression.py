# Compresses all images in a directory into JPEG-2000s at three different compression levels: 1000, 500, and 100.
import os
import cv2

# Path to the directory containing the Tiny-Imagenet images
tiny_imagenet_path_train = './tiny-imagenet/train'

# Path to the directories for the three different compression levels
compression_1000_path_train = './jpeg-compressions/level-1000/train'
compression_500_path_train = './jpeg-compressions/level-500/train'
compression_100_path_train = './jpeg-compressions/level-100/train'

# # List of all class directories in the Tiny-Imagenet train dataset
# classes = os.listdir(tiny_imagenet_path_train)

# # Loop through all class directories
# for class_name in classes:
#     # Path to the class directory
#     class_path = os.path.join(tiny_imagenet_path_train, class_name, 'images')
    
#     # List of all images in the class directory
#     images = os.listdir(class_path)
    
#     # Loop through all images in the class directory
#     for image_name in images:
#         # Path to the image
#         image_path = os.path.join(class_path, image_name)

#         # Read the image
#         image = cv2.imread(image_path)

#         # Compress the image at three different compression levels
#         cv2.imwrite(os.path.join(compression_1000_path_train, image_name), image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000])
#         cv2.imwrite(os.path.join(compression_500_path_train, image_name), image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X500])
#         cv2.imwrite(os.path.join(compression_100_path_train, image_name), image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X100])

# Path to the directory containing the Tiny-Imagenet test images
tiny_imagenet_path_test = './tiny-imagenet/test/images'

# Path to the directories for the three different compression levels
compression_100_path_test = './jpeg-compressions/level-100/test'
compression_50_path_test = './jpeg-compressions/level-50/test'
compression_10_path_test = './jpeg-compressions/level-10/test'

# List of all images in the Tiny-Imagenet test dataset
images = os.listdir(tiny_imagenet_path_test)

# Loop through all images in the test dataset
for image_name in images:
    # Path to the image
    image_path = os.path.join(tiny_imagenet_path_test, image_name)

    # Read the image
    image = cv2.imread(image_path)

    # Compress the image at three different compression levels
    cv2.imwrite(os.path.join(compression_100_path_test, image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    cv2.imwrite(os.path.join(compression_50_path_test, image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 30])
    cv2.imwrite(os.path.join(compression_10_path_test, image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 10])

print('All images compressed successfully!')


