import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = mpimg.imread('exit-ramp.jpg')
plt.imshow(image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')

# Include Gaussian smoothing, which is essentially a way of suppressing noise and spurious gradients by averaging.
# http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur
# Note: cv2.Canny() actually applies Gaussian smoothing internally,
#       but we include it here because you can get a different result by applying further smoothing.
kernel_size = 5     # Must be an odd number (3, 5, 7...)
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# Note: As far as a ratio of low_threshold to high_threshold,
#       John Canny himself recommended a low to high ratio of 1:2 or 1:3
#       http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html#steps
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(gray, low_threshold, high_threshold)
plt.imshow(edges, cmap='Greys_r')

plt.show()
