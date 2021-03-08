# vehicle-color-recognition-using-CNN-model

## Description
> In this project, we have used Convolutional Neural Network (CNN) architecture for detecting vehicle color. The colors are used in this project:
> Beige, Black, Blue, Brown, Gray, Green, orange, Red, Silver, White, Yellow.

>Hint: Dataset and training files are not available, Until the paper related this project will be published. Meanwhile, you are able to utilize test models for predicting vehicle color by using model weights named `color_model.h5` and prediction code in Google Colab named `color_prediction.ipynb`
>

>Creating an input file named `input` containing images.


## Road map for predicting vehicle color:
> First step: Mount your google colab drive by using below instructions:
```
from google.colab import drive
drive.mount('/content/drive/')
import os
os.chdir("/content/drive/MyDrive/train python/IMAGE-AI")
!ls
```
> Second step: Install requirements in google colab
```
!pip install -r requirements.txt
```

> Third step: Importing necessary libraries
```
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
```
> Fourth Step: Identifying input size and defining list of colors

```
img_width, img_height = 224, 224
CATEGORIES=['beige','black','blue', 'brown', 'gray' ,'green', 'orange','red', 'silver', 'white','yellow']
```

> Fifth step: load model weights
```
model = load_model('C:/Users/payagostar/Desktop/color_model.h5')

```

> Last step: after builing input image file, run the below instructions:
```
path="/content/drive/MyDrive/train python/IMAGE-AI/input"
x=os.listdir(path)
print(x)
for img in x:
  test_image = image.load_img(os.path.join(path,img), target_size=(img_width, img_height,3))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis=0)
  test_image = test_image.reshape(1,img_width, img_height,3)
  result = model.predict_classes(test_image, batch_size=1)
  print("this car in "+img+"is:", CATEGORIES[result[0]])
  print("---------------------")
```

## Result 

![Capture1](https://user-images.githubusercontent.com/77263576/110308409-50036380-8015-11eb-858d-ded63f35b59c.PNG)

