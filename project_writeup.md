# **Behavioral Cloning**

## Project Writeup


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.png "Model Architecture"
[image2]: ./examples/center.png "Center Image"
[image3]: ./examples/left.png "Left Image"
[image4]: ./examples/right.png "Right Image"
[image5]: ./examples/recovery.png "Recovery Image"
[image6]: ./examples/not_flipped.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"
[image8]: ./examples/Figure.png "Results"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* project_writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 136-159)

The model includes RELU layers to introduce nonlinearity (code lines 139, 141, 143, 145, and 146), and the data is normalized in the model using a Keras lambda layer (code line 137).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 140, 142, and 144).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 133, 134). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 155).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and augmenting provided images to provide generalized situations.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the resources from the course, in combination with collecting my own training data.

My first step was to use a convolution neural network model similar to the NVIDIA model and only the resources from the course (images). I thought this model might be appropriate because it is what the course used to pepare students for the project. After further exploration, the NVIDIA model was adjusted slightly to perform better in this given situation.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model (the original NVIDIA model) had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to add dropout layers between the early convolution layers.

Since the original model was still behaving poor in specific portions of the track, I decided to collect more data manually. This included driving the car manually around the track twice, while also collecting "recovery" data from the side of the road. This recovery data was collected by moving the car off of the road, starting to record, moving the vehicle back into the center of the road, and pausing the recording. This was done many times throughout the track and added to the data set.

While these images certainly helped the model drive the car around the track successfully, I found the car tended to weave back and forth across the road. I decided then to augment the images to help it generalize. These augmentations included flipping images (remove left turn bias of track 1), random translation in the X and Y axis, masking the image with a shadow (all credit for function implemenation to Naokishibuya as provided in model.py), altering the brightness of the image, and converting the RGB image to YUV for the NVIDIA model as discussed in the Slack channels.

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 136-159) consisted of a convolution neural network with the following layers and layer sizes (as seen in the visualization below).

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Images were also recorded from the left and right side of the vehicle as shown below.

![alt text][image3]
![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself before hitting the edge of the road. This image shows an example of where the car starts from for recovery data collection.

![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would remove the left turn bias from the track itself. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

As discussed above, the images were also subject to random translation (Slack channel discussion led to this decision), shadowing an image (masking, source and credit to Naokishibuya), altering the brightness through HSV manipulation, and converting to the YUV color space before feeding it into the model.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the figure below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image8]

Overall, the car was able to drive autonomously on both courses. You'll notice from the second video (on track 2) that recodring is cut short. Unfortunately, my workstation was not able to continue recording data due to lack of resources (sadly it's an old machine). While I could have skipped uploading the track 2 video completely (as it's not part of the rubric), I wanted to show that this pipeline and approach to this project is fully functional on both tracks, given your workstation has decent hardware. Further work to collect more data, especially recovery data, would be helpful. Likewise, I'd like to further expand the augmentation process to help avoid collecting data manually.