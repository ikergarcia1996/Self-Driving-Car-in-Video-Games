# T.E.D.D. 1104
A supervised deep neural network that learns to drive in video games. The main objective of this project is to 
achieve a model that can drive in Grand Theft Auto V. The model is not expected to drive following traffic 
regulations, but imitate how humans drive in this game: Drive at full speed through the city avoiding other 
cars and occasionally humans and lampposts.
A marker will be set in the game map, the model should be able to arrive to the marker driving trough the city. 

The model is trained using human labeled data. We record the game and key inputs of humans while the play the game, this data
is used to train the model. 

While we focus on self-driving cars and the video game Grand Theft Auto V this model can be adapted to play any existing
video game. 
    <table>
    <tr>
    <td> <img src="github_images/demo.gif" alt="gotta go fast!"/> </td>
    <td> <img src="github_images/demo2.gif" alt="gotta go fast2!"/> </td>
    </tr>
    </table>


## Table of Contents
- [T.E.D.D. 1104](#tedd-1104)
  * [Table of Contents](#table-of-contents)
  * [1) News](#1-news)
  * [2) Architecture](#2-architecture)
  * [3) Software and HOW-TO](#3-software-and-how-to)
    + [3.1) Requirements](#31-requirements)
    + [3.2) Generate dataset](#32-generate-dataset)
    + [3.3) Train the model](#33-train-the-model)
      - [Improving the robustness of the model](#improving-the-robustness-of-the-model)
    + [3.4) Run the model](#34-run-the-model)
      - [Pretrained models](#pretrained-models)
      - [Extra features](#extra-features)
  * [4) Authors:](#4-authors)


## 1) News
**NEW 15/04/2020** First pretrained model released!!, [click here to download it](https://github.com/ikergarcia1996/Self-Driving-Car-in-Video-Games/releases/tag/0.2). For instructions on how to run the AI see the [Run the Model](#34-run-the-model) Section  
**NEW 7/04/2020** Let's generate a huge training dataset together!! [Click here so see how to collaborate in the project (Spanish)](https://youtu.be/utQoMGLbCFc). 
**We have reached 1,5Tb of training data (~150 hours of gameplay)!!!!**  

## 2) Architecture
Most previous models that attempt to achieve self-driving in video games consists of a deep convolutional neural network 
(usually Inception or Resnet). The models take as input a single image.
 Would you be able to know what to do if I give you this image?
 
<p align="center">
  <img src="github_images/choices.png" alt="Make a choice"/>
</p>

You may think that the best choice is to brake to avoid the blue/gray car, but, 
what if both cars are stopped waiting for you to cross the street? What if your car is 
currently driving in reverse? Does your current speed and the speed of the other cars allow you to 
cross the road without hitting them? A single image does not provide enough information to successfully 
achieve a self-driving car. More information is needed, that is why our approach uses sequences of images. 
5 images are captured with an interval of 1/10 seconds between them, this approach gives the model information 
about the motion of other cars, environment and himself. 


<p align="center">
  <img src="github_images/sequence.png" alt="Sequences 4 the win"/>
</p>

T.E.D.D. 1104 follows the End-to-end (E2E) learning approach and it consists of a Deep Convolutional Neural Network (Resnet: K He et al. 2016) 
followed by a Recurrent Neural Network (LSTM). The CNN receives as input a sequence of 5 images and generates for each one a 
vector representation. These representations are fed into the RNN that generates a unique vector representation 
for the entire sequence. Finally, a Feed-Forward Neural Network outputs the key to press in the keyboard based 
on the vector representation for the sequence.

<p align="center">
  <img src="github_images/network_architecture.png" alt="The brain!"/>
</p>

The model has been implemented using Pytorch: https://pytorch.org/

## 3) Software and HOW-TO
This repository contains all the files need for generating the training data, training the model and use the model to 
drive in the video game. The software has been written in Python 3. This model has only been tested in Windows 10 because
is the only supported SO by most video games.  

### 3.1) Requirements
```
Pytorch (1.4.0 or newer reccomended)
Torchvision
numpy
glob
h5py
json 
cv2 (opencv-python)
win32api (PythonWin) - Should be installed by default in newest Python versions for Windows (Python 3.7 reccomended)
cupy (optional but highly recommended, 10x speed up in data preprocessing comparated with numpy)
Nvidia Apex (only for using FP16)
tensorboard (only for training a model)
```

### 3.2) Generate dataset 
* File: generate_data.py
* Usage example: 
```
python generate_data.py --save_dir tedd1007\training_data
```
* How-to:
  * Set your game in windowed mode
  * Set your game to 1600x900 resolution
  * Move the game window to the top left corner, there should be a blue line of 1 pixel in the left bezel of your
         screen and the window top bar should start in the top bezel of your screen.
  * Play the game! The program will capture your screen and generate the training examples. There will be saved
         as files named "training_dataX.npz" (numpy compressed array). Don't worry if you re-launch this script,
          the program will search for already existing dataset files in the directory and it won't overwrite them.
  * At any moment push Q + E to stop the program.
  
<p align="center">
  <img src="github_images/example_config.png" alt="Setup Example"/>
</p>
  


### 3.3) Train the model 
* File: train.py
* Usage example: 
```
python train.py --train_new 
--train_dir tedd1007\training_data\train 
--dev_dir tedd1007\training_data\dev 
--test_dir tedd1007\training_data\test 
--output_dir tedd1007\models\model1
--batch_size 10 
--num_epochs 5 
--fp16
```
* How-to:
  Train a model using the default hyper parameters, to see a description of the network hyper parameters use 
  "python train.py -h" or check the "train.py" and "model.py" files. train, dev and test directories should contain
   as many files named "training_dataX.npz" as you want. The FP16 flag allows you to use Mixed Precision Training if
   you have a modern Nvidia GPU with Tensor cores (RTX 2000, RTX Titan, Titan V, Tesla V100...), 
   it uses the Nvidia Apex library: https://github.com/NVIDIA/apex.
   The model is VERY memory demanding, as a
   reference I use a batch size of 15 for a RTX 2080 (8GB VRAM) for FP16 training (half the Vram usage than FP32 training) 
   using the default parameters. 
   
 * If you want to continue training from a checkpoint use (Note: The checkpoint will automatically use the same 
 floating point precision (FP16 or FP32) used for training when it was created):
   
 ```
python train.py --continue_training
--train_dir tedd1007\training_data\train 
--dev_dir tedd1007\training_data\dev 
--test_dir tedd1007\training_data\test 
--output_dir tedd1007\models 
--batch_size 10 
--num_epochs 5 
--checkpoint_path tedd1007\models\model1\checkpoint.pt
```

#### Improving the robustness of the model
As every other neural network, TEDD1104 tries to find the easiest way of replicating the training examples. 
* TEDD1104 will tend to focus on the in-game minimap, this will result in a model that is very good following the roads in mini-map but ignores other cars or obstacles. To avoid that "--hide_map_prob" parameter sets a probability of removing (put a black square) the minimap from all the images of a training example. I recommend using a value between 0.4 and 0.5. 
* Removing (black image) some of the images from an input sequence, especially the last one, can also help to improve 
 the robustness of the model. If one of the images of the sequence is removed, TEDD1104 will be forced to "imagine" that 
 image, improving the trajectory prediction capabilities of the model. It will also force the model to use the 
 information from all the images in the sequence instead of relying on the last one. We can set a probability for 
 removing each input image for a training example with the parameter --dropout_images_prob followed by 5 floats. 
 Using a bidirectional LSTM can also be useful. 
* Scheduler:  --scheduler_patience allows setting a number of iterations. If the loss function does not decrease 
 after the specified number of iterations the learning rate is reduced (new_learning_rate = learning rate * 0.1). 
 This helps to further improve the model after the loss function stops decreasing. 
* Gradient accumulation: TEDD1104 is very memory demanding. In my RTX 2080 (8GB VRAM) using FP16, I can only set a
 batch size between 10 and 20 which might be too low. To increase the batch size you can use gradient accumulation.
 Gradient accumulation allows increasing the batch size without increasing the VRAM usage. You can set the number of
 batches to accumulate with the parameter --gradient_accumulation_steps. The effective batch size will equal
 --batch_size * --gradient_accumulation_steps. 
* Validation data: The best validation data (dev and test) are files of routes through the map driving different 
vehicles and driving in different weather conditions (including day/night). DO NOT USE as dev or test set random examples
are taken from the training set because they will be part of a sequence of similar data, that is, a high dev and test accuracy
will correspond to an overfitted model. Note that we save the model that achieves the highest accuracy in the dev test.
* Since the training data is generated recoding humans driving, each training file will store a sequence of continuous examples, that is,
similar weather conditions, the same vehicle, a lot of similar training examples... To improve the robustness of the model 
it would be ideal to shuffle the entire training dataset. When you have a very big dataset shuffling all the examples can
take many days or even weeks (+1TB data). An alternative is loading multiple random files (i.e. 5) during training and 
shuffling the examples of the loaded files. The parameter --num_load_files_training sets the number of files that will be loaded
and shuffled. The higher the value, the higher RAM usage. 
This is an example of a command for training a small model taking into all the described improvements into account.
```
python train.py --train_new 
--train_dir tedd1007\training_data\train 
--dev_dir tedd1007\training_data\dev 
--test_dir tedd1007\training_data\test 
--output_dir tedd1007\models\small+
--batch_size 20 
--gradient_accumulation_steps 4 
--num_epochs 5 
--num_load_files_training 5
--optimizer_name SGD 
--learning_rate 0.01 
--scheduler_patience 100 
--bidirectional_lstm 
--dropout_lstm_out 0.2 
--dropout_images_prob 0.2 0.2 0.2 0.2 0.3 
--hide_map_prob 0.4
--fp16 
```
  
During training you can use tensorboard to visualize the loss and accuracy:
```
tensorboard --logdir='./runs'
```

### 3.4) Run the model

* File: run_TEDD1104.py
* Usage example: 
```
python run_TEDD1104.py --model_dir tedd1007\models\model1 --show_current_control --fp16
```

Use the FP16 flag if you have an Nvidia GPU with tensor cores (RTX 2000, RTX Titan, Titan V...) 
for a nice speed up (~x2 speed up) and half the VRAM usage. 
Requires the Nvidia Apex library: https://github.com/NVIDIA/apex


* How-to:
  * Set your game in windowed mode
  * Set your game to 1600x900 resolution
  * Move the game window to the top left corner, there should be a blue line of 1 pixel in the left bezel of your
         screen and the window top bar should start in the top bezel of your screen.
  * Let the AI play the game!
  * Push Q + E to exit
  * Push L to see the input images
  * Push and hold J to use to use manual control
          
<p align="center">
  <img src="github_images/example_config.png" alt="Setup Example"/>
</p>

#### Pretrained models:
Pretrained models are available in the releases section: [Releases sections](https://github.com/ikergarcia1996/Self-Driving-Car-in-Video-Games/releases/)

#### Extra features 
* By default, the model will record a sequence of images with an interval of 0,1secs between each image. 
This means that the model will predict a key to push 10 times per second (every time the sequence is updated). 
You can increase this value with the --num_parallel_sequences parameter. num_parallel_sequences=2 means that 20 
sequences per second will be recorded (2 sequences will be recorded in parallel updating them every 0,05sec), 
num_parallel_sequences=3 30... Recoding more sequences per second can help TEDD1104 to drive better, 
but a faster CPU and memory will be required to ensure a 0,1sec delay between each image in all the sequences. 
A warning will be printed if the CPU is not able to update the sequences fast enough. 
Using an i7 8700K I can record up to 2 sequences.
* The model may crash into a wall, car or other obstacle and be unable to return to the road.
 The model implements an "evasion manoeuvre", if the first and the last images in a sequence of 
 images are very similar (i.e car is stuck facing a wall) it will automatically drive backwards for 1 
 second and then randomly turn left or right for 0,2 seconds. To enable this feature use the --enable_evasion 
 flag and select the sensitivity to trigger the evasion manoeuvre (the difference between images calculated using 
 mean squared error) with the --evasion_score parameter (default 200). Note that this option requires to calculate 
 the mean squared error between two images each iteration, so it will increase the time the model needs to process 
 an input sequence.


  

## 4) Authors:
```
- Iker García
  Personal Webpage: https://ikergarcia1996.github.io/Iker-Garcia-Ferrero/
```

This repository is a greatly improved version of the model we published 2 years ago: https://github.com/ikergarcia1996/GTAV-Self-driving-car (by Eritz Yerga and Iker García)
  
