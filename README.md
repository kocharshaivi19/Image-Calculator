README.md (Shaivi Kochar)

○ Describe the steps in your pipeline, what choices did you take and why

○ The layers in your model, why did you pick them

○ What performance are you currently seeing, why is that

○ How to train your system (preferably just running one or two files)

○ How to run your system (preferably just running one or two files)


In Response:

-------------------------------------------------------------------------------------------------------------------------------------
PIPELINE
-------------------------------------------------------------------------------------------------------------------------------------
1. Firstly, I have manually colleced dataset of Mathematical Operators {'+', '-', '*', 'x'}that are missing in the dataset.
The dataset have similar specifications as MNIST dataset : GRAYSCALED Image with shape (28,28)

2. Saving MNIST training dataset and Mathematical Operators dataset in HDF5 file which is readable by Caffe Layer

3. Validation Dataset: Using few OpenCV techniques on the given Equation Dataset to extract equations and its corresponding parts (digits and operators). Validation Dataset includes 
	- Equation wise segmented digits and operators Images.
		- Resizing the given image to the size of (500, 500)
		- Deskewing using cv2.HoughLines and cv2.WarpAffine Transformation
		- Column wise Image Segmentation to extract the subparts
		- Resizing all the corresponding image to (28, 28) with a pixel patch of (20,20)
	- Text file that maps each image to its true label. 

4. Intermediate Model: This Deep Learning Framework on Caffe models, trains, validats and recognises the Digits and Operators.
	- Initialising Parameters and Layers Caffe. I have used Pre-defined MNIST layer architecture.
	- Traning and Validating the Arithmatical MNIST combo dataset to store the weights in the caffemodel. 

5. Final Model: This model is an Abstract Syntax Tree to generate the result of the Equation.
	- Storing Predicted Labels for each digit and operator in the Equation from intermediate model.
	- The function check for the relevency between consecutive Opeartor, Operand for AST nodes and replaces them with the returned value of the method.

6. Generating the Prediction and Accuracy of Intermediate model and Final Model to depict the strength of the system.

-----------------------------------------------------------------------------------------------------------------------------------------
FUNTIONS DEFINATION
-----------------------------------------------------------------------------------------------------------------------------------------
|-model.py ************************* (Main File to Run the System) *********************************
|-opencvutils.py ******************* (Preprocessing OpenCV Funtions) *******************************
|-lenetcaffe.py ******************** (Caffe Model / Intermediate Model) ****************************
|-evaluation.py ******************** ( AST Model / Final Model) ************************************
|-caffemodel
	|-lenet_train.prototxt ********* (Network Layer Parameters) ************************************
	|-lenet_deploy.prototxt ******** (Testing Parameters) ******************************************
	|-lenet_solver.prototxt ******** (Training and Validation Parameters for Stochastic Gradient)***
|-data
	|-hdf5
		|-mnist_train_data.hdf5 **** (HDF5 Training Dataset) ***************************************
		|-mnist_val_data.hdf5 ****** (HDF5 Validation Dataset) *************************************
		|-train.txt **************** (Train file to Network Layer for Data Tracking) ***************
		|-val.txt ****************** (Val file to Network Layer for Data Tracking) *****************
|-dataset
	|-{Raw Validation Image dataset}
	|-data.csv
|-augumented
	|-{Manually generated Arithmatic Operators Image Dataset}
|-digits
	|-{Specific Folder for Equation / Equation Wise folders}
|-mldata
	|-mnist-original.mat *********** (.Mat file of the MNIST data from mldata repo) ****************
|-README.md
|-requirements.txt


----------------------------------------------------------------------------------------------------------------------------------------
LAYERS IN THE INTERMEDIATE MODEL
----------------------------------------------------------------------------------------------------------------------------------------
I thought of Augumenting my own data for Operators because of scarcity of desirable dataset. 
Preprocessing is required to extract the essential parts of the dataset as we have MNIST dataset for digits only. 
Training, Validation and Testing through Deep Learning enhances the prediction. Also, I am using the same Layer Architechute as MNIST solely to compare the Accuracy and Prediction over the MNIST. The combine result of all subparts is fed to Final Model to evalue the equation generating a better chances 

----------------------------------------------------------------------------------------------------------------------------------------
PERFORMANCE
----------------------------------------------------------------------------------------------------------------------------------------
1. Generation of Validation Dataset: 
	1.1 Considering noises like Rotation and Skewness of the Equation in the image, I am able to augment a clean, deskew data.  
	1.2 Segmenting of Individual Digits or Operator from the Equation is not generating good results bacause of the two reasons:
		1.2.1. Since, we need to maintain the sequence of the subparts in the equation, Opencv technique to detect contours may not line in a sqeuence. Though, it was able to detect most of them.
		1.2.2. My technique to segment the subparts is able to preserave the sequence and its Prediction can easily be traced from Intermediate Model.
2. Intermediate Model Performance:
	1.1 Accuracy
3. Final Model Performace:
	1.1 It is dependent on the Intermediate Model Prediction, in itself it is relient to generate correct results.

----------------------------------------------------------------------------------------------------------------------------------------
ASSUMPTIONS
----------------------------------------------------------------------------------------------------------------------------------------
1. Pre-Installation of Caffe
2. Pre-Installation of OpenCV
3. Pre-Installation of HDF5 

----------------------------------------------------------------------------------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------------------------------------------------------------------------------
1. Please paste dataset into the respective folder.
2. Change the root dir path in model.py to desired location.
3. Run: python model.py to run the system