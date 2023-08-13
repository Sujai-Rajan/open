# Final Project

# Author: Sujai Rajan
# CS-6140 Machine Learning

# Description: This file is used to train the model using the dataset generated from the simulator.


print('\nInitializing Code...\n')
## Importing the functions from functions.py
from functions import *
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To suppress the warnings 

# Step 1 - Importing the dataset
path = 'Sim_Data_Final/'
img_dir = path + 'IMG/'
data = import_dataset_info(path)
print('Dataset Imported\n')


## Step 2 - Visualizing the dataset 
n_bins = 31
samples_per_bin = 1000
balance_dataset(data, n_bins, samples_per_bin, display=False)
print('Dataset Balanced\n')


## Step 3 - Preprocessing the dataset
image_path, steering_angle = load_data(path, data)
print('Dataset Loaded\n')


## Step 4 - Splitting the dataset into training and validation set
test_size_given = 0.2
random_state_given = 7  # User defined parameters
image_path_train, image_path_val, steering_angle_train, steering_angle_val = training_validation_split(image_path, steering_angle, test_size_given, random_state_given)
print('Dataset Splitted\n')


## Step 5 - Augmenting the dataset

# The function is made to augmnet 6 functions in random order of 50% probability
# Function is applied to the training set only by the batch generator function


## Step 6 - Preprocessing the dataset

# The function is made to crop the image, change the color space to YUV, apply Gaussian Blur, resize and normalize the image
# Function is applied to both training and validation set by the batch generator function


## Step 7 - Creating batches of the datasetpi
batch_size_given = 32
# batch_generator_train = batch_generator(image_path_train, steering_angle_train, batch_size_given, True)      # Training set batch generator 
# batch_generator_val = batch_generator(image_path_val, steering_angle_val, batch_size_given, False)           # Validation set batch generator with augmentation set to False



# ## Step 8 - Creating the model
# model = create_model()
# model.summary()    # Summary of the model
# print('Model Created\n')

## Step 8.1 - Loading the model
model = load_model('mac_chck_best_model0.03967.h5')
print('Model Loaded\n')


## Step 8.2 - Compiling the model
model.compile(loss='mse', optimizer='adam')
print('Model Compiled\n')



## Step 9 - Training the model
from keras.callbacks import ModelCheckpoint
epochs_given = 10
steps_per_epoch_given = 200
history = model.fit((batch_generator(img_dir, image_path_train, steering_angle_train, batch_size_given, True)), steps_per_epoch=steps_per_epoch_given, epochs=epochs_given, max_queue_size = 1 , validation_data=(batch_generator(img_dir, image_path_val, steering_angle_val, batch_size_given, False)), validation_steps= 100, verbose=True, callbacks = [ModelCheckpoint('mac_chck_best_model{val_loss:.5f}.h5', monitor='val_loss', verbose=True, save_best_only=True, mode='auto')])
print('Model Trained\n')


# Step 10 - Saving the model
model.save('mac_win_autonomous_model.h5')
print('Model Saved\n')


# Step 11 - Plotting the results
plot_results(history)
print('Results Plotted\n')


# Step 12 - Saving the plot
plt.savefig('mac_win_results.png')
print('Results Saved\n')








