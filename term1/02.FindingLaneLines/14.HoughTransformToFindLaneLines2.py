# coding=utf-8
import numpy as np
import cv2

# Note: Understanding Hough Transform
#       https://alyssaq.github.io/2014/understanding-hough-transform/

image = cv2.imread('exit-ramp.jpg')
#cv2.imshow("original", image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#cv2.imshow("gray", gray)

# Include Gaussian smoothing, which is essentially a way of suppressing noise and spurious gradients by averaging.
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
#cv2.imshow("blur gray", blur_gray)

# Define parameters for Canny and run it
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(gray, low_threshold, high_threshold)
#cv2.imshow("edges", edges)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color = 255

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)
#vertices = np.array([[(0, imshape[0]), (imshape[1] / 2, imshape[0] / 2), (imshape[1], imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
#cv2.imshow("masked edges", masked_edges)

# Let's look at the input parameters for HouthLinesP
# 'rho' and 'theta' are the distance and angular resolution of our grid in Hough space.
# Remember that, in Hough space, we have a grid laid out along the (Θ, ρ) axis.
# You need to specify 'rho' in units of pixels and 'theta' in units of radians.
#
# So, what are reasonable values? Well, rho takes a minimum value of 1,
# and a reasonable starting place for theta is 1 degree (pi/180 in radians).
# Scale these values up to be more flexible in your definition of what constitutes a line.
#
# 'threshold' specifies the minimum number of votes(intersections in a given grid cell)
#   a candidate line needs to have to make it into the output.
# 'np.array[]' is just a placeholder, no need to change it.
# 'min_line_length' is the minimum length of a line (in pixels) that you will accept in the output.
# 'max_line_gap' is the maximum distance (in pixels) between segments
#   that you will allow to be connected into a sigle line.
rho = 2                 # distance resolution in pixels of the Hough grid
theta = np.pi/180       # angular resolution in radians of the Hough grid
threshold = 15          # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40    # minimum number of pixels making up a line
max_line_gap = 20       # maximum gap in pixels between connectable line segments
# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

# Create a blank to draw lines on
line_image = np.copy(image) * 0

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
cv2.imshow("line image", line_image)

threshold = 50
lines2 = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
line2_image = np.copy(image) * 0
for line in lines2:
    for x1, y1, x2, y2 in line:
        cv2.line(line2_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
cv2.imshow("line2 image", line2_image)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))

# Draw the lines on the edge image
line_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
cv2.imshow("line edges", line_edges)

key = cv2.waitKey(0)
if key == 27:   # wait for ESC
    cv2.destroyAllWindows()
