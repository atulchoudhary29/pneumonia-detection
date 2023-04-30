import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from skimage.transform import resize
from skimage import measure
from matplotlib import patches
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

# Define a function that calculates the Intersection over Union (IoU) loss between two tensors
def iou_loss(y_true, y_pred):
    # Cast the input tensors to float32 data type
    y_true=tf.cast(y_true, tf.float32)
    y_pred=tf.cast(y_pred, tf.float32)
    
    # Reshape the tensors into vectors
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
   
    # Calculate the intersection between the two tensors by element-wise multiplication and summation
    intersection = tf.reduce_sum(y_true * y_pred)
    
    # Calculate the score using the intersection, sum of the true tensor, and sum of the predicted tensor
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    
    # Return the IoU loss by subtracting the score from 1
    return 1 - score


# Define a function that combines binary cross-entropy (BCE) loss and IoU loss
def iou_bce_loss(y_true, y_pred):
    # Calculate the BCE loss and IoU loss between the true and predicted tensors
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    
    # Combine the losses with equal weights
    loss = 0.5 * bce + 0.5 * iou
    
    # Return the combined loss
    return loss


# Define a function that calculates the mean Intersection over Union (IoU) as a metric between two tensors
def mean_iou(y_true, y_pred):
    # Round the predicted tensor to convert it to binary values
    y_pred = tf.round(y_pred)
    
    # Calculate the intersection between the true and predicted tensors by element-wise multiplication and summation
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    
    # Calculate the union between the true and predicted tensors by summation
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    
    # Add a smoothing factor to prevent division by zero
    smooth = tf.ones(tf.shape(intersect))
    
    # Calculate the mean IoU by averaging the IoU score for each image in the batch
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))


# Define a function that loads and preprocesses an image from a file
def load_and_preprocess_image(filename, image_size=128):
    # Load the image from file using the PIL library
    img = Image.open(filename)
    
    # Convert the image to a NumPy array
    img = np.array(img)
    
    # Resize the image to the specified size using the scikit-image library
    img = resize(img, (image_size, image_size), mode='reflect')
    
    # Add a channel dimension to the image
    img = np.expand_dims(img, -1)
    
    # Add a batch dimension to the image
    img = np.expand_dims(img, 0)
    
    # Return the preprocessed image
    return img


# Import the `load_model` function from Keras to load a saved model from a file
from keras.models import load_model

# Load the trained model from the saved file and specify the custom loss functions
model = load_model('model_new.h5', custom_objects={'iou_bce_loss': iou_bce_loss, 'mean_iou': mean_iou})

# Define a function that loads an image from file using the file dialog
def load_image():
    # Declare global variables to store the image and file path
    global img, img_path
    
    # Use the file dialog to select an image file and check if a file was selected
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    
    # Update the image path variable and load the image using PIL
    img_path.set(file_path)
    img = Image.open(file_path).resize((256, 256))
    
    # Convert the image to a Tkinter-compatible PhotoImage and display it in a label
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img


# Define a function that deletes the currently loaded image and prediction
def delete_image():
    # Declare the global image variable
    global img
    
    # Clear the image label and image path variable
    img_label.config(image='')
    img_label.image = None
    img_path.set('')
    
    # Clear the predicted image label and prediction and coordinates text variables
    img_label_pred.config(image='')
    img_label_pred.image = None
    prediction_text.set('')
    coordinates_text.set('')


# Define a function that loads and preprocesses the image, makes a prediction, and displays the results
def predict():
    # Check if an image is loaded and return if not
    if not img_path.get():
        return
    
    # Load and preprocess the image, and make a prediction using the trained model
    img = load_and_preprocess_image(img_path.get())
    pred = model.predict(img)

    # Apply threshold to the predicted mask to obtain a binary mask
    pred_mask = pred[0, :, :, 0] > 0.5

    # Plot the original and predicted images side by side using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100, facecolor='none')
    ax.imshow(img[0, :, :, 0], cmap='gray')

    # Apply connected components and bounding boxes to the predicted mask, and calculate the percentage of area affected
    comp = measure.label(pred_mask)   # Label the connected components in the binary mask using scikit-image
    bbox_count = 0                   # Initialize a count of the number of bounding boxes
    total_area = 0                   # Initialize a variable to store the total area affected
    img_area = img.shape[1] * img.shape[2]   # Calculate the total area of the image
    coordinates_string = ""          # Initialize a string to store the coordinates of each bounding box
    for region in measure.regionprops(comp):   # Loop over the connected components and extract their properties
        y, x, y2, x2 = region.bbox   # Get the bounding box coordinates
        height = y2 - y              # Calculate the height of the bounding box
        width = x2 - x               # Calculate the width of the bounding box
        area = height * width        # Calculate the area of the bounding box
        total_area += area           # Add the area of the bounding box to the total area affected
        bbox_count += 1              # Increment the count of the number of bounding boxes
        coordinates_string += f"Coordinates: x={x}, y={y}, x2={x2}, y2={y2}\n"   # Add the bounding box coordinates to the string
        ax.add_patch(patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none'))   # Draw the bounding box on the plot using matplotlib patches

    affected_percentage = (total_area / img_area) * 100   # Calculate the percentage of area affected by dividing the total area by the image area and multiplying by 100


    # Update the prediction and coordinates text variables based on the prediction results
    prediction_text.set(f"Percentage of area affected: {affected_percentage:.2f}%")   # Update the prediction text variable to display the percentage of area affected by the pneumonia
    if bbox_count == 0:   # If no bounding boxes were detected, update the condition label to display "Normal" in green font
        prediction_label.config(fg='green')
        condition = "Normal"
    else:   # Otherwise, update the condition label to display "Pneumonia" in red font
        prediction_label.config(fg='red')
        condition = "Pneumonia"

    coordinates_text.set(f"{coordinates_string}Condition: {condition}")   # Update the coordinates text variable to display the coordinates of the bounding boxes and the condition
    if condition == "Normal":   # If the condition is "Normal", update the coordinates label to display "Normal" in green font
        coordinates_label.config(fg='green')
    else:   # Otherwise, update the coordinates label to display "Pneumonia" in red font
        coordinates_label.config(fg='red')


    # Customize the plot and save it as an image, then display it in the GUI
    ax.set_title('')   # Remove the title of the plot
    ax.axis('off')   # Remove the axis labels and ticks from the plot
    fig.canvas.draw()   # Redraw the canvas to update the plot
    width, height = fig.canvas.get_width_height()   # Get the width and height of the canvas
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, -1)   # Convert the canvas buffer to a numpy array of RGBA values
    pred_img = Image.fromarray(buf, mode='RGBA')   # Convert the numpy array to a PIL Image object
    pred_img = pred_img.resize((512, 256))   # Resize the image to fit the GUI
    pred_img = ImageTk.PhotoImage(pred_img)   # Convert the PIL Image object to a PhotoImage object that can be displayed in the GUI
    img_label_pred.config(image=pred_img)   # Update the predicted image label to display the PhotoImage
    img_label_pred.image = pred_img
    plt.close(fig)   # Close the matplotlib figure to free up memory

# Create the main window
window = tk.Tk()   # Create a new Tkinter window object
window.title("Pneumonia Detection")   # Set the title of the window
window.geometry("1024x600")   # Set the size of the window

# Global variables
img = None   # Initialize a variable to store the loaded image
img_path = tk.StringVar()   # Create a Tkinter StringVar object to store the path to the loaded image
prediction_text = tk.StringVar()   # Create a Tkinter StringVar object to store the prediction text
coordinates_text = tk.StringVar()   # Create a Tkinter StringVar object to store the coordinates text

# Create and position widgets
button_frame = tk.Frame(window)   # Create a new Tkinter frame object to hold the buttons
button_frame.pack(side=tk.TOP, pady=10)   # Pack the frame into the top of the window with some padding

load_button = tk.Button(button_frame, text="Load Image", command=load_image)   # Create a new Tkinter button object to load an image
load_button.pack(side=tk.LEFT, padx=10)   # Pack the button into the left side of the button frame with some padding

delete_button = tk.Button(button_frame, text="Delete Image", command=delete_image)   # Create a new Tkinter button object to delete an image
delete_button.pack(side=tk.LEFT, padx=10)   # Pack the button into the left side of the button frame with some padding

predict_button = tk.Button(button_frame, text="Predict", command=predict)   # Create a new Tkinter button object to predict pneumonia
predict_button.pack(side=tk.LEFT, padx=10)   # Pack the button into the left side of the button frame with some padding

center_frame = tk.Frame(window)   # Create a new Tkinter frame object to hold the images
center_frame.pack(side=tk.TOP, pady=30)   # Pack the frame into the top of the window with some padding

img_label = tk.Label(center_frame)   # Create a new Tkinter label object to display the loaded image
img_label.pack(side=tk.LEFT, padx=100)   # Pack the label into the left side of the center frame with some padding

img_label_pred = tk.Label(center_frame)   # Create a new Tkinter label object to display the predicted image
img_label_pred.pack(side=tk.RIGHT, padx=100)   # Pack the label into the right side of the center frame with some padding

prediction_label = tk.Label(window, textvariable=prediction_text, font=("Arial", 14))   # Create a new Tkinter label object to display the prediction text
prediction_label.pack(side=tk.BOTTOM, pady=50)   # Pack the label into the bottom of the window with some padding

coordinates_label = tk.Label(window, textvariable=coordinates_text, font=("Arial", 14))   # Create a new Tkinter label object to display the coordinates text
coordinates_label.pack(side=tk.BOTTOM, pady=20)   # Pack the label into the bottom of the window with some padding

# Start the main loop
window.mainloop()   # Start the main loop of the Tkinter window to display the GUI and handle user input
