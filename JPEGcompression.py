# Compresses all images in a directory into JPEG-2000s at three different compression levels: 1000, 500, and 100.
import os
import cv2

# Path to the directory containing the Tiny-Imagenet images
tiny_imagenet_path = 'INSERT_TINY_IMAGENET_PATH_HERE'

# Path to the directories for the three different compression levels
compression_1000 = 'INSERT_COMPRESSION_1000_PATH_HERE'
compression_500 = 'INSERT_COMPRESSION_500_PATH_HERE'
compression_100 = 'INSERT_COMPRESSION_100_PATH_HERE'

# List of all images in the directory
images = os.listdir(tiny_imagenet_path)

# Loop through all images in the directory
for image in images:
    # Load the image
    img = cv2.imread(tiny_imagenet_path + image)
    
    # Save the image at three different compression levels
    cv2.imwrite(compression_1000 + image, img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000])
    cv2.imwrite(compression_500 + image, img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X500])
    cv2.imwrite(compression_100 + image, img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X100])

