# Generating Abstract Art using DC_GAN
### In this project I have generated abstract by training DC_GAN model with aprrox 8500 paintings

<img src="results.gif"/>

You can generate any kind of figure using this

### to use this 

make sure each image have same **height x width**

```python
INPUT_DATA_DIR = "/tf/DC_GAN/art_dataset/" # Path to the folder with your inputs
```

### Tunning parameters

Change the parameters according to your need

```python
# Hyperparameters
NOISE_SIZE = 100
LR_D = 0.00004
LR_G = 0.0004
BATCH_SIZE = 64
EPOCHS = 300
BETA1 = 0.5
WEIGHT_INIT_STDDEV = 0.02
EPSILON = 0.00005
SAMPLES_TO_SHOW = 5
```
### Tunning OpenCV

Change the interpolations in openCV eg.RGB,HSV

```pythoninput_images = np.asarray([np.asarray(cv2.cvtColor(cv2.resize(cv2.imread(file),(128,128)), cv2.COLOR_BGR2RGB)) for file in glob(INPUT_DATA_DIR + '*')])
```
