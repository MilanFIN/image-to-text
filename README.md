# image-to-text
Text detection from images with pytorch

### dependencies
`pip3 install torch opencv-python numpy`
* pytorch might require a specific version if using on a machine with no nvidia gpu, see: https://pytorch.org/get-started/locally/


### how to use

* run the program for an image that has been placed under the images folder (example.jpg used here)
```
python3 imagetext.py example.jpg
```
Example of the execution for included test image:

<img src="https://github.com/MilanFIN/image-to-text/blob/main/images/example.jpg?raw=true" alt="image with sample text" width="300"/>

```
This is example
text
Thsting l2456789
There will be some
errors
```

* training a new pair of neural nets (default models are under models/ folder)
```
python3 train.py
```

### how it works

* The program first loads the image, and besides a few tricks thresholds it to get the text to be visible from the background.
* Lines and individual characters are separated, and fed into a neural network to be classified into letters & digits
* The program keeps track of average distances between characters to separate words
* Two separate neural networks are used. The larger one is used to classify first letter of each word and the smaller one for other letters. The larger one is predicting out of both capital and lowercase letters, and the small one is only predicting out of lowercase ones. This was done as the smaller network has better accuracy and words usually don't have capital letters in the middle.
* The networks are trained by a batch of generated sample images that can be requested from functions in libs/createimages.py
