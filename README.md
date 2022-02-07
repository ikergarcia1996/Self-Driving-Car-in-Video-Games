# T.E.D.D. 1104
A supervised deep neural network that learns how to drive in video games. The main objective of this project is to 
achieve a model that can drive in Grand Theft Auto V. Given a waypoint, the model is expected to reach the destination as
fast as possible avoiding other cars, humans and obstacles. 

The model is trained using human labeled data. We record the game and key inputs of humans while they play the game, this data
is used to train the model. 

While we focus on self-driving cars and the video game Grand Theft Auto V this model can be adapted to play any existing
video game. 

<table>
<tr>
<td> <img src="github_images/demo.gif" alt="gotta go fast!"/> </td>
<td> <img src="github_images/demo2.gif" alt="gotta go fast2!"/> </td>
</tr>
</table>

# Pretrained T.E.D.D. 1104 models
We provide pretrained T.E.D.D. 1104 models that you can use for real-time inference :)  
The models are trained using 130 GB of human labelled data.  
The model has been trained in first-person-view with a route to follow in the minimap.  
The model has learned to drive a large variety of vehicles in different weather conditions (Sun, night, sunny, rain...). 

### T.E.D.D. 1104 Base: [Num_params] parameters. 
Download link: See the Releases Tab

Accuracy in the test datasets:

|         |              Time              |   Weather  | Micro-Acc K@1 | Micro-Acc k@3 | Macro-Acc K@1 |
|---------|:------------------------------:|:----------:|:-------------:|:-------------:|:-------------:|
| City    |         :sun_with_face:        |   :sunny:  |      49.8     |      83.8     |      44.1     |
| City    |         :sun_with_face:        | :umbrella: |      52.1     |      84.7     |      46.1     |
| City    | :first_quarter_moon_with_face: |   :sunny:  |      54.5     |      86.9     |       48      |
| City    | :first_quarter_moon_with_face: | :umbrella: |      48.8     |      82.5     |      43.2     |
| Highway |         :sun_with_face:        |   :sunny:  |      65.6     |      100      |      53.2     |
| Highway |         :sun_with_face:        | :umbrella: |      70.6     |       98      |      54.2     |
| Highway | :first_quarter_moon_with_face: |   :sunny:  |      71.3     |      100      |      52.3     |
| Highway | :first_quarter_moon_with_face: | :umbrella: |      67.7     |      100      |      50.9     |

### T.E.D.D. 1104 Large: 
Coming Soon...
###  T.E.D.D. 1104 Small:
Coming Soon...

# Datasets
We provide train/dev/test datasets for training and evaluating T.E.D.D 1107 models:
- Train Dataset (~130Gb): Coming soon... 
- Dev Dataset (~495Mb): [Download Dev+Test datasets](https://drive.google.com/file/d/1SutVGsQKg0mDUkfGML1nBboLWi5e5_4E/view?usp=sharing).
- Test Dataset (~539Mb): [Download Dev+Test datasets](https://drive.google.com/file/d/1SutVGsQKg0mDUkfGML1nBboLWi5e5_4E/view?usp=sharing).


##  Architecture

T.E.D.D. 1104 follows the End-To-End learning approach. We approach the task as a classification task. 
The input of the model is a sequence of 5 images, each image has been recorded with a 0.1s interval. 
The output are the correct keys in the keyboard to press. Alternatively T.E.D.D. 1104 can also be trained  
with a regression objective using xbox controller inputs. 

<p align="center">
  <img src="github_images/network_architecture.png" alt="The brain!"/>
</p>

The model consists of three modules:
First a **Convolutional Neural Network** that encodes each input image in a feature 
vector. We use EfficientNet (https://arxiv.org/abs/1905.11946).
We use a **transformer encoder** (https://arxiv.org/abs/1706.03762) to generate bidirectional joint distributions over the feature vector
sequence. Finally, we use the [CLS] token to predict the key combination. 

The model has been implemented using Pytorch: https://pytorch.org/ and PyTorch Lightning: https://www.pytorchlightning.ai/

# Software and HOW-TO
This repository contains all the files need for generating the training data, training the model and use the model to 
drive in the video game. The software has been written in Python 3. Model can be trained in any OS. Data generation
and inference only works in Windows 10/11 which are the only SO supported by most video games. 

## Requirements
```
Python 3.7 or newer (3.9.7 tested)
Pytorch 1.6.0 or newer (1.10 tested)
Torchvision (0.11.0 tested)
PyTorch Lightning (1.4.9 tested)
torchmetrics
numpy
PIL/Pillow
cv2 (opencv-python)
json
tkinter
tabulate
win32api (PythonWin) - Should be installed by default in newest Python versions for Windows 

pygame - Only required if you wish to generate data using a Xbox Controller
PYXInput - Only required if you wish to use a Vitual Xbox Controller as game controller instead of the keyboard. 
           See controller\setup.md for installation instructions. 
```

## Train your own model
### Self Driving Model

Use the *train.py* script to train a new model from scratch or continue training a model. 
See "train.py -h" to get a description of all the available parameters. 

Example command:
```
Sample command
```

You can continue training a model using the "--continue_training" flag 
```
Sample command
```

#### Evaluate model:
Use the eval.py script to evaluate a model in the test dataset.
```
Sample command
```

### Image Reordering Model
An experimental unsupervised pretraining objective. We shuffle the order of the input sequence and the model must 
predict the correct order of the input images. See "train_reorder.py -h" to get a description of all the available parameters. 
After training with the image reordering objective you can finetune the model in the Self-Driving objective. 

```
python3 train.py \
--new_model \
--checkpoint_path models/image_reordering.ckpt \
...
```

Use the eval.py script to evaluate a image reordering model in the test dataset.



## Run Inference 
How to use a pretrained T.E.E.D. 1104 model to drive in GTAV

### Configure the game
You can run the game in "windowed mode" or "full screen" mode. 
If you want to run the game in "windowed mode":
- Run GTAV and set your game to windowed mode
- Set your the desired game resolution (i.e 1600x900 resolution)
- Move the game window to the top left corner
- Run the script with the "--width 1600" and "--height 900" parameters

<p align="center">
  <img src="github_images/example_config.png" alt="Setup Example"/>
</p>
  

If you want to run the game in "full screen" mode:
- Run GTAV **in your main screen** (The one labelled as screen nº1) and set your game to full screen mode
- Configure the game resolution with the resolution of your screen (i.e 2560x1440 resolution)
- Run the script with the "--width 2560", "--height 1440" and "--full_screen"

<p align="center">
  <img src="github_images/example_config_full_screen.png" alt="Setup Example Full Screen"/>
</p>

In addition, if you want to run the pretrained models that we provide you must:
- Set the Settings>Camera>First person Vehicle Hood to "On"
- Change the camera to first-person-view (Push "V")
- Set a waypoint in the minimap

<p align="center">
  <img src="github_images/additional_config.jpg" alt="Setup Example"/>
</p>


### Run a Model

Use the *run_TEDD1104.py* script to run a model for real-time inference. See "run_TEDD1104.py -h" to get a description of all the available parameters. 

```
python run_TEDD1104.py \
--checkpoint_path "models\TEDD1107_model.ckpt" \
--width 1920 \
--height 1080 \
--num_parallel_sequences 5 \
--control_mode keyboard
```
num_parallel_sequences: number of parallel sequences to record, if the number is higher the model will do more 
iterations per second (will push keys more often) provided your GPU is fast enough. This improves the performance of the 
model but increases the CPU and RAM usage. 

control_mode: Choose between keyboard and controller (Xbox Controller). It doesn't matter how the model has been trained, 
the output of the model will be converted to the desired control_mode. 

If the model does not perform as expected (It doesn't seem to do anything or always chooses the same action) you can 
push "L" while the script is running to verify the input images. 


## Generate Data

Use the *generate_data.py* script to generate new data for training. See Use "run_TEDD1104.py -h" to get a description of all the available parameters.
Configure the game following [The Configure the game section](#configure-the-game). 
```
python generate_data.py \
--save_dir "dataset/train" \
--width 1920 \
--height 1080 \
--control_mode keyboard
```

If control_mode is set to "keyboard" we will record the state of the "WASD" keys. If control_mode is set to "controller"
we will record the state of the first Xbox Controller that Pygame can detect. 
To avoid generating a huge unbalanced dataset the script will try to balance the data while recording. The more examples
of a given class recorded the lower the probability of recording a new example of that class. If you want 
to disable this behaviour use the "--save_everything" flag. 


## Citation:
```
@misc{TEDD1104,
  author = {"Garc{\'\i}a-Ferrero, Iker},
  title = {TEDD1104: Self Driving Car in Video Games},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ikergarcia1996/Self-Driving-Car-in-Video-Games}},
}
```

Author: **Iker García-Ferrero**:  
- [My Webpage](https://ikergarcia1996.github.io/Iker-Garcia-Ferrero/)  
- [Twitter](https://twitter.com/iker_garciaf)

