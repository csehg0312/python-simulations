import numpy as np
import cv2
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the image
input_image = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)
if input_image is None:
    raise ValueError("Could not open or find the image.")

# Normalize the image to [0, 1]
input_image = input_image.astype(float) / 255.0

# Step 2: Define the neighborhood and templates
neighborhood = np.ones((3, 3))  # 3x3 neighborhood
A = np.array([[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]])       # Identity template
B = np.array([[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]])    # Edge detection template

# Step 3: Implement fuzzy membership functions
x = np.linspace(0, 1, 100)
low = fuzz.gaussmf(x, 0.25, 0.1)
medium = fuzz.gaussmf(x, 0.5, 0.1)
high = fuzz.gaussmf(x, 0.75, 0.1)

# Step 4: Define the fuzzy CNN update rule
def fuzzy_cnn_update(pixel_value, neighbors, A, B):
    # Apply A and B templates
    a_product = np.sum(A * neighbors)
    b_product = np.sum(B * neighbors)
    
    # Fuzzify the pixel value
    mu_low = fuzz.interp_membership(x, low, pixel_value)
    mu_medium = fuzz.interp_membership(x, medium, pixel_value)
    mu_high = fuzz.interp_membership(x, high, pixel_value)
    
    # Apply fuzzy rules
    if mu_low > 0.5:
        output_value = 0.0  # Set low pixels to black
    elif mu_high > 0.5:
        output_value = 1.0  # Set high pixels to white
    else:
        output_value = pixel_value  # Keep medium pixels unchanged
    
    return output_value

# Step 5: Iterate over the image pixels
rows, cols = input_image.shape
output_image = np.zeros_like(input_image)

# Pad the image to handle edges
padded_image = np.pad(input_image, ((1,1),(1,1)), mode='symmetric')

for i in range(rows):
    for j in range(cols):
        # Extract the neighborhood
        neighbors = padded_image[i:i+3, j:j+3]
        # Get the current pixel value
        current_pixel = input_image[i, j]
        # Update the pixel value using the fuzzy CNN rule
        output_image[i, j] = fuzzy_cnn_update(current_pixel, neighbors, A, B)

# Step 6: Post-processing and visualization
# Amplify changes
output_image = output_image * 2.0
output_image = np.clip(output_image, 0, 1)

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title('Processed Image')
plt.axis('off')

plt.show()

# Highlight differences
difference = np.abs(output_image - input_image)
plt.imshow(difference, cmap='gray')
plt.title('Difference')
plt.show()