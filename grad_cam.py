import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, toimage, imsave
from imagenet_classes import class_names
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from vgg16 import vgg16
import double2image

class GradCam:
    @ops.RegisterGradient("GuidedRelu")
    def _GuidedReluGrad(op, grad):
        return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))
        
if __name__ == '__main__':
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        sess = tf.Session()
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16(imgs, 'vgg16_weights.npz', sess, 242)

        img1 = imread('cat_dog.jpg', mode='RGB')
        img1 = imresize(img1, (224, 224))
        ll = sess.run([vgg.probs, vgg.heatmap], feed_dict={vgg.imgs: [img1]})
        prob = ll[0][0]
        #heatmap = np.array(ll[1])
        heatmap = ll[1]
        #print heatmap

        # gradCam = np.array(heatmap[0][0]) # [7, 7, 512]
        # print np.shape(gradCam)
        # gradCam = toimage(gradCam)
        # #heatmap = double2image.to_image(heatmap)
        # gradCam = imresize(gradCam, (224, 224))
        # imsave('gradCam.png', gradCam)

        # gb = np.array(heatmap[1][0])   # [7, 7, 512]
        # print np.shape(gb)
        # gb = toimage(gb)
        # #heatmap = double2image.to_image(heatmap)
        # gb = imresize(gb, (224, 224))
        # imsave('gb.png', gb)

        guidedGradCam = np.array(heatmap[2][0]) # [224, 224, 3]
        print np.shape(guidedGradCam)
        guidedGradCam = toimage(guidedGradCam)
        #heatmap = double2image.to_image(heatmap)
        guidedGradCam = imresize(guidedGradCam, (224, 224))
        imsave('guidedGradCam.png', guidedGradCam)

        #print np.shape(prob)
        #print np.shape(heatmap)
        #print heatmap
        #preds = (np.argsort(prob)[::-1])[0:5]
        #for p in preds:
        #    print p, class_names[p], prob[p]