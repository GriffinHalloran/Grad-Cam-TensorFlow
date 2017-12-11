# Grad-Cam-TensorFlow
This is a code reimplementation of Grad-Cam code in TensorFlow for the paper

**[Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization][1]**  
Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, Dhruv Batra  
[https://arxiv.org/abs/1610.02391][1]

original code in Caffe: [https://github.com/ramprs/grad-cam][2]

![Overview](http://i.imgur.com/JaGbdZ5.png)

### Usage
#### VGG 16
Download VGG16 Tensorflow model weight [vgg16_weights.npz][3]

Run ``` python grad_cam.py ```

To change input image and category, modify line 23 and 24
```
    category = 977
    img_file = 'Failure Images/failure4.png'
```
List of categories can be found in [imagenet_classes.py][4]. Note that the first line is not an actual category. To use the category of interest, find the category in this file and use ```line_num-1``` as input.

#### Other Model
To explore Grad-CAM with other model and input image (e.g. Cifar-10), use the iPython notebook [cifar-10.ipynb][5].

[1]: https://arxiv.org/abs/1610.02391
[2]: https://github.com/ramprs/grad-cam
[3]: http://www.cs.toronto.edu/~frossard/post/vgg16/
[4]: https://github.com/vivianbuan/Grad-Cam-TensorFlow/blob/master/imagenet_classes.py
[5]: https://github.com/vivianbuan/Grad-Cam-TensorFlow/blob/master/cifar-10.ipynb
