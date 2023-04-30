# Pneumonia Detection

This project is a deep learning model for detecting pneumonia in chest x-rays. It uses the RSNA Pneumonia Detection Challenge dataset, which contains over 26,000 chest x-ray images labeled as "normal" or "pneumonia".

The model is built using the TensorFlow and Keras libraries, and the graphical user interface (GUI) is built using the Tkinter library.

## Table of Contents

1. [Dataset](#dataset)
2. [Converting DICOM to Images](#dicom)
3. [Installation Dependencies](#dependencies)
4. [Usage](#usage)


 <a name="dataset"></a> 
## 1. Dataset

- The dataset used for this project is the RSNA Pneumonia Detection Challenge on Kaggle. Please download the dataset and extract the contents to the dataset folder in the project directory.

	https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

 <a name="dicom"></a> 
## 2. Converting DICOM to Images

- Before training the model, the DICOM images need to be converted to JPEG images. This can be done using the dcm_to_jpg.py script provided in the project directory. The script takes one argument: the conversion type, which should be either "train" or "test".

- The converted images will be saved in the dataset/images or dataset/samples directory, depending on the conversion type.

 <a name="dependencies"></a> 
## 3. Dependencies

- The required Python packages for this project are listed in requirements.txt. They can be installed using pip:

	```
	pip install -r requirements.txt
	```

<a name="usage"></a>  
## 4. Usage
  
1. Graphical User Interface

	- To use the GUI, run the gui.py script
	- The GUI will open, and you can use the "Load Image" button to select a chest x-ray image from your computer. Once an image is loaded, you can click the "Predict" button to run the image through the trained model and see the results. The GUI will display the original image with any detected pneumonia areas highlighted in red, as well as a prediction text indicating the percentage of the image affected by pneumonia and the condition of the image ("Normal" or "Pneumonia"). If there are multiple areas of pneumonia in the image, the GUI will display the coordinates of each area.

2. Jupyter Notebook

	- To run the notebook, open a terminal in the project directory and type:

		```
		jupyter notebook
		```

	- The model.ipynb notebook contains the code for training the pneumonia detection model. It can be run in a Jupyter Notebook environment such as JupyterLab or Google Colab. The notebook includes code for loading and preprocessing the image data, building and training the model, and evaluating the performance of the model on a validation set. The notebook also includes code for saving the trained model to a file for use in the GUI.
