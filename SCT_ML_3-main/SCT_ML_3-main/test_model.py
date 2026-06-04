# ================================
# test_model.py
# User selects image using file dialog
# ================================

import cv2
import joblib
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load Saved Model
model = joblib.load("cats_dogs_svm_model.pkl")

IMG_SIZE = 64

# Hide main tkinter window
Tk().withdraw()

# Open file dialog
img_path = askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

# Check if user selected image
if not img_path:
    print("No image selected!")
    exit()

# Read image
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check image
if image is None:
    print("Invalid image!")
    exit()

# Resize image
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

# Flatten image
image = image.flatten().reshape(1, -1)

# Predict
prediction = model.predict(image)

# Result
if prediction[0] == 0:
    result = "Cat Detected"
else:
    result = "Dog Detected"

print(result)

# Show image with result
display_img = cv2.imread(img_path)

cv2.putText(
    display_img,
    result,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Prediction", display_img)

cv2.waitKey(0)
cv2.destroyAllWindows()