## Final Course Project - Autonomous Driving Simulation

## Sujai Rajan

## CS-6140 Machine Learning


# Importing the libraries
import pandas as pd                                         # for importing the dataset
import numpy as np                                          # for mathematical operations                         
import matplotlib.pyplot as plt                             # for plotting the graphs  
import os                                                   # for accessing the files
from sklearn.utils import shuffle                           # for shuffling the dataset      
from sklearn.model_selection import train_test_split        # for splitting the dataset into training and validation set
import matplotlib.image as mpimg                            # for importing the images (to work with RGB format over BGR in OpenCV)
from imgaug import augmenters as iaa                        # for image augmentation
import cv2                                                  # for image processing
import random                                               # for random number generation
from tensorflow.keras.models import Sequential              # for creating the model (Sequential model- Linear stack of layers)
from tensorflow.keras.layers import Convolution2D,Flatten,Dense,Lambda,Dropout # for creating the layers of the model (Convolution2D- for convolutional layers, Flatten- for flattening the layers, Dense- for fully connected layers), Lambda- for normalization, Dropout- for regularization
from tensorflow.keras.optimizers.legacy import Adam                # for optimizing the model  (Adam- Adaptive Moment Estimation - Adaptive learning rate optimization algorithm)
from tensorflow.keras.callbacks import ModelCheckpoint     # for saving the model (ModelCheckpoint- to save the model after every epoch)










# Function to get name of the image file
def get_filename(path):
    filename = path.split('\\')[-1]
    return filename
















#  Function to import the dataset information
def import_dataset_info(path):
    column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(path + 'driving_log.csv', names=column_names)
    print('\nDataset has been imported\n')
    print('\nShape of the dataset is: \n', data.shape)

    print('\nFirst 5 rows of the dataset are: \n')
    print(data.head()) # print the first 5 rows of the data

    print('\nSummary of the dataset is: \n')
    print(data.describe())  # print the summary of the data

    # # Vefify function get_filename
    # print(data['center'][0])
    # print(get_filename(data['center'][0]))

    data['center'] = data['center'].apply(get_filename)  # apply function get_filename to the column 'center'
    data['left'] = data['left'].apply(get_filename)  # apply function get_filename to the column 'left'
    data['right'] = data['right'].apply(get_filename)  # apply function get_filename to the column 'right'

    print('\n(post changing names)First 5 rows of the dataset are: \n')
    print(data.head())  # print the first 5 rows of the data

    return data









# Function to balance the dataset (Essetial to avoid overfitting and to improve the performance of the model and to avoid bias)
def balance_dataset(data, n_bins, samples_per_bin, display=True):            # display=True to display the histogram or False to not display
    n_bins = n_bins                                     # number of bins  
    samples_per_bin = samples_per_bin                           # number of samples per bin (Depending on Histogram Plot)

    # Histogram of the steering angles before balancing
    hist_st, bins_st = np.histogram(data['steering'], n_bins)          # histogram of the steering angles

    
    ('\nBins of the steering angles before balancing are: \n', bins_st)

    # To display the histogram for finding the samples_per_bin
    # Flag to display the histogram
    if display:

        center = (bins_st[:-1] + bins_st[1:]) * 0.5                       # centering the bins

        # Balancing the dataset
        print(center)

        # Plotting the histogram of the steering angles
        plt.bar(center, hist_st, width=0.01)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
        plt.title('Histogram of the steering angles before balancing')
        plt.show()


    # Removing the data points from the bins with more than samples_per_bin data points
    remove_list_index = []

    # Looping for each bin
    for j in range(n_bins):
        bin_data_list = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins_st[j] and data['steering'][i] <= bins_st[j + 1]:
                bin_data_list.append(i)

        # Shuffle the data points (so points removed wont be from the same place)
        bin_data_list = shuffle(bin_data_list)
        bin_data_list = bin_data_list[samples_per_bin:]
        remove_list_index.extend(bin_data_list)         # extend the list of data points to be removed

    # Print length of the list of data points to be removed and to be kept and total
    print('\nTotal number of data points: ', len(data))
    print('\nNumber of data points to be removed: ', len(remove_list_index))
    print('\nNumber of data points to be kept: ', len(data) - len(remove_list_index))

    # Drop the data points to be removed
    data.drop(data.index[remove_list_index], inplace=True)  


    # Flag to display the histogram
    if display:
        hist_st,_= np.histogram(data['steering'], n_bins)          # histogram of the steering angles

        # Plotting the histogram of the steering angles
        plt.bar(center, hist_st, width=0.01)
        plt.plot((-1,+1), (samples_per_bin, samples_per_bin))
        plt.title('Histogram of the steering angles after balancing')
        plt.show()
    








## Function to load all the data (center, left, right images and steering angles)
def load_data(path, data):

    X = []  # images
    y = []  # steering angles

    X= data[['center', 'left', 'right']].values

    y = data['steering'].values

    

    # Verify the data
    print('\nFirst 5 rows of the images are: \n', X[:5])
    print('\nFirst 5 rows of the steering angles are: \n', y[:5])

    return X, y








# Function to split the dataset into training and validation set using sklearn
def training_validation_split(image_path, steering_angle, test_size_given, random_state_given):
    
    image_path_train, image_path_val, steering_angle_train, steering_angle_val = train_test_split(image_path, steering_angle, test_size=test_size_given, random_state=random_state_given) # test_size=0.2 means 20% of the data will be used for validation set

    # Print the shape of the arrays
    print('\nTotal number of images in the dataset: ', len(image_path))
    print('\nTotal number of images in the training set: ', len(image_path_train))
    print('\nTotal number of images in the validation set: ', len(image_path_val))

    return image_path_train, image_path_val, steering_angle_train, steering_angle_val











# Function to choose the image from the center, left or right camera
def choose_image(center, left, right, steering_angle, steering_correction):
        
        choice = np.random.choice(3) # choose a random number from 0, 1 or 2
    
        # 0 - Center Image
        if choice == 0:
            return mpimg.imread(os.path.join(center)), steering_angle
    
        # 1 - Left Image
        elif choice == 1:
            return mpimg.imread(left), steering_angle + steering_correction
    
        # 2 - Right Image
        else:
            return mpimg.imread(right), steering_angle - steering_correction














# Function to augment the data
def augment_dataset(center, left, right, steering_angle):



    steering_correction = 0.2 # steering correction for the left and right camera images
    img = 0.0

    img, steering_angle = choose_image(center, left, right, steering_angle, steering_correction) # choose an image randomly from the center, left or right camera



    ## Data Augmentation Steps

    if np.random.rand() < 0.5:                                                                # 50% chance of applying the following augmentation

        # 1. Pan the image (Translation)
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})            # pan the image         x = 0.1 means 10% of the image width and y = 0.1 means 10% of the image height
        img = pan.augment_image(img)                                                        # augment the image with pan

    if np.random.rand() < 0.5:                                                                # 50% chance of applying the following augmentation

        # 2. Zoom the image
        zoom = iaa.Affine(scale=(1, 1.2))                                                   # zoom the image         1.2 means zoom in by 20%
        img = zoom.augment_image(img)                                                       # augment the image with zoom

    if np.random.rand() < 0.5:                                                                # 50% chance of applying the following augmentation

        # 3. Change the brightness of the image
        brightness = iaa.Multiply((0.2, 1.2))                                               # change the brightness of the image     0.2 means reduce brightness by 20% and 1.2 means increase brightness by 20%
        img = brightness.augment_image(img)                                                 # augment the image with brightness

    if np.random.rand() < 0.5:                                                                # 50% chance of applying the following augmentation

        # 4. Flip the image
        img = cv2.flip(img, 1)                                                               # flip the image horizontally
        steering_angle = -steering_angle                                                      # reverse the steering angle for the flipped image

    if np.random.rand() < 0.5:                                                                # 50% chance of applying the following augmentation

        # 5. Change the contrast of the image
        contrast = iaa.GammaContrast(gamma=(0.2, 1.8))                                      # change the contrast of the image      0.2 means reduce contrast by 20% and 1.8 means increase contrast by 80%
        img = contrast.augment_image(img)                                                   # augment the image with contrast


    if np.random.rand() < 0.3:                                                                # 50% chance of applying the following augmentation

        # 6. Change the shear of the image
        shear = iaa.Affine(shear=(-7, 7))                                                   # change the shear of the image          -7 means shear counter-clockwise by 7 degrees and 7 means shear clockwise by 7 degrees
        img = shear.augment_image(img)                                                      # augment the image with shear



    return img, steering_angle









# Function to preprocess the images from the dataset
def preprocess_dataset(image):

    # Crop the image
    image = image[60:-25, :, :]                                                             # crop the image to remove the sky and the car bonnet  (crop the image from 60th pixel to 135th pixel in the y-axis)

    # Convert the image from RGB to YUV                                             
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)                                          # YUV is better than RGB as it is more efficient in detecting the edges of the road (Suggested by NVIDIA)

    # Resize the image
    image = cv2.resize(image, (200, 66), cv2.INTER_AREA)                                    # resize the image to 200x66 (width x height)

    return image









# Function to generate the batch of images and steering angles
def batch_generator(img_dir, image_path, steering_angle, batch_size, Training_flag):
    
    # Initializing the lists to store the images and steering angles
    image_batch = []
    steering_angle_batch = []

    while True:                                                                             # while loop is used to generate the batch of images and steering angles continuously
        
        for i in range(batch_size):

            # Generate a random index number within the range of the dataset
            random_index = random.randint(0,(len(image_path)-1))

            # Get the image and steering angle from the random index
            center, left, right = image_path[random_index]
            # Add image path 
            center = img_dir + center
            left = img_dir + left
            right = img_dir + right
            steering = steering_angle[random_index]

            # Augment the image and steering angle if training is set to True
            if Training_flag and np.random.rand() < 0.6:
                image, steering = augment_dataset(center, left, right, steering)

            else:
                image = mpimg.imread(center)                          # read the image from the directory
                steering = steering
            
            # Preprocess the image
            image = preprocess_dataset(image)

            # Append the image and steering angle to the list
            image_batch.append(image)
            steering_angle_batch.append(steering)


        # Convert the lists to numpy arrays
        yield(np.asarray(image_batch), np.asarray(steering_angle_batch))                    # yield is used to return a generator object (it is similar to return but it returns a generator object instead of a value)









# Create the model
def create_model():                                                                                 # Elu activation function is used instead of Relu as it avoids the dying relu problem (when the relu function outputs 0 for all the negative inputs) Negative Gradient Problem (when the gradient is negative for all the negative inputs) and Zero-Centered Activation Problem (when the mean of the output is not zero) 
                                                                                
    # Sequential model is used to add layers one by one                                               
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66, 200, 3)))                              # Normalize the image (divide by 127.5 and subtract 1.0) and input shape is 66x200x3 (height x width x depth)

    # Add the layers to the model
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))         # 24 filters of size 5x5 with stride of 2x2 and input shape of 66x200x3 (height x width x depth) and activation function is elu

    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))                                   # 36 filters of size 5x5 with stride of 2x2 and activation function is elu

    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))                                   # 48 filters of size 5x5 with stride of 2x2 and activation function is elu

    model.add(Convolution2D(64, (3, 3), activation='elu'))                                           # 64 filters of size 3x3 with activation function is elu (Stride is not specified so it is 1x1 by default as the shape of image is small)

    model.add(Convolution2D(64, (3, 3), activation='elu'))                                           # 64 filters of size 3x3 with activation function is elu (Stride is not specified so it is 1x1 by default as the shape of image is small)

    model.add(Dropout(0.5))                                                                          # Dropout layer with dropout rate of 0.5 (50% of the neurons are dropped)
    
    model.add(Flatten())                                                                             # Flatten the output of the convolutional layers

    model.add(Dense(100, activation='elu'))                                                          # Fully connected layer with 100 neurons and activation function is elu

    model.add(Dense(50, activation='elu'))                                                           # Fully connected layer with 50 neurons and activation function is elu

    model.add(Dense(10, activation='elu'))                                                           # Fully connected layer with 10 neurons and activation function is elu

    model.add(Dense(1))                                                                              # Fully connected layer with 1 neuron (output layer)  Predicted steering angle is the output of this layer

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')                                  # Adam optimizer is used with learning rate of 0.0001 and mean squared error is used as the loss function

    # MSE is used as the loss function instead of cross entropy because it is a regression problem and not a classification problem as we are predicting a continuous value (steering angle) and not a discrete value (class)
    # Adam optimizer is used instead of SGD (Stochastic Gradient Descent) because it is more efficient than SGD as it uses momentum and adaptive learning rate (it uses different learning rates for different parameters) and it converges faster than SGD

    return model                            









# Function to Plot the results
def plot_results(history):

    # Plotting the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    print('Training and Validation Loss Plotted\n')

    

