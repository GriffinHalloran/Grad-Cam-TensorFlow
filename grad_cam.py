import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, toimage, imsave
from skimage.transform import resize
from imagenet_classes import class_names
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from vgg16 import vgg16
import double2image
from utils import GradCam, Guided_backprop

class GuidedRelu:
    @ops.RegisterGradient("GuidedRelu")
    def _GuidedReluGrad(op, grad):
        return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

if __name__ == '__main__':
    #-------
    #gradcam
    #-------
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    category = 977
    img_file = 'Failure Images/failure4.png'
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess, category)
    img1 = imread(img_file, mode='RGB')
    img1 = imresize(img1, (224, 224))
    gradcam = GradCam(vgg.probs, vgg.pool5, category)
    tf.summary.FileWriter('tf.log', sess.graph)
    ll1 = sess.run([vgg.probs, gradcam], feed_dict={vgg.imgs: [img1]})
    prob = ll1[0][0]
    heatmap = np.array(ll1[1])
    #heatmap = toimage(heatmap)
    heatmap = double2image.to_image(heatmap)
    heatmap = imresize(heatmap, (224, 224))
    imsave('gradcam.png', heatmap)
    #---------------
    #Guided backprop
    #---------------
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        sess = tf.Session()
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16(imgs, 'vgg16_weights.npz', sess, category)
        img1 = imread(img_file, mode='RGB')
        img1 = imresize(img1, (224, 224))
        guided_backprop = Guided_backprop(vgg.imgs, vgg.probs, category)
        ll = sess.run([vgg.probs, guided_backprop], feed_dict={vgg.imgs: [img1]})
        print ll[1].shape
        guided_backprop = toimage(ll[1])
        #heatmap = double2image.to_image(heatmap)
        #guidedGradCam = imresize(guided_backprop, (224, 224))
        imsave('guided_backprop.png', guided_backprop)
    #---------------
    #Guided gradcam
    #---------------
    guided_gradcam = ll[1]
    cam = resize(ll1[1]/np.max(ll1[1]), (224, 224), preserve_range = True)
    for i in range(3):
        guided_gradcam[:, :, i] = guided_gradcam[:, :, i] * cam
    imsave('guided_gradcam.png', toimage(guided_gradcam))
        #preds = (np.argsort(prob)[::-1])[0:5]
        #for p in preds:
        #    print p, class_names[p], prob[p]
