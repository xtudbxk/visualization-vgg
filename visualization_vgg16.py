import os
import sys

import numpy as np
from skimage import io
from skimage import transform as transf
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Zoomout_Vgg16:
    def __init__(self, vgg16_npy_path=None,zlayers=["conv1_1","conv1_2","conv2_1","conv2_2","conv3_1","conv3_2","conv3_3","conv4_1","conv4_2","conv4_3","conv5_1","conv5_2","conv5_3","relu6","relu7"],downsample=4,weight=224,height=224,deconv_layer="pool5"):
        self.zlayers = zlayers
        self.zlayers_num = len(self.zlayers)
        self.net = {}
        self.strides={"conv1_1":1,"conv1_2":1,"pool1":2,
                     "conv2_1":2,"conv2_2":2,"pool2":4,
                     "conv3_1":4,"conv3_2":4,"conv3_3":4,"pool3":8,
                     "conv4_1":8,"conv4_2":8,"conv4_3":8,"pool4":16,
                     "conv5_1":16,"conv5_2":16,"conv5_3":16,"pool5":32,
                    }
        self.channels={"conv1_1":64,"conv1_2":64,"pool1":64,
                     "conv2_1":128,"conv2_2":128,"pool2":128,
                     "conv3_1":256,"conv3_2":256,"conv3_3":256,"pool3":256,
                     "conv4_1":512,"conv4_2":512,"conv4_3":512,"pool4":512,
                     "conv5_1":512,"conv5_2":512,"conv5_3":512,"pool5":512,
                    }
        self.downsample = downsample
        self.w = weight
        self.h = height
        self.w_d = int(weight / downsample)
        self.h_d = int(height / downsample)
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        self.build()
        self.net["input_deconv"] = tf.placeholder(shape=[1,int(224/self.strides[deconv_layer]),int(224/self.strides[deconv_layer]),self.channels[deconv_layer]],dtype=tf.float32)
        self.net["output_deconv"] = self.build_deconv(this_layer=deconv_layer,feature_maps=self.net["input_deconv"])

    def build(self):
        self.net["input"] = tf.placeholder(shape=[1,self.w,self.h,3],dtype=tf.float32)

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.net["input"])
        assert red.get_shape().as_list()[1:] == [self.w, self.h, 1]
        assert green.get_shape().as_list()[1:] == [self.w, self.h, 1]
        assert blue.get_shape().as_list()[1:] == [self.w, self.h, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [self.w, self.h, 3]

        self.net["conv1_1"] = self.conv_layer(bgr, "conv1_1")
        self.net["conv1_2"] = self.conv_layer(self.net["conv1_1"], "conv1_2")
        self.net["pool1"] = self.max_pool(self.net["conv1_2"], 'pool1')
        tmp = tf.tile(self.net["pool1"],[1,1,2,2])
        tmp = tf.reshape(tmp,self.net["conv1_2"].shape)
        self.net["pool1_mask"] = tf.cast(tf.greater_equal(self.net["conv1_2"],tmp),dtype=tf.float32)

        self.net["conv2_1"] = self.conv_layer(self.net["pool1"], "conv2_1")
        self.net["conv2_2"] = self.conv_layer(self.net["conv2_1"], "conv2_2")
        self.net["pool2"] = self.max_pool(self.net["conv2_2"], 'pool2')
        tmp = tf.tile(self.net["pool2"],[1,1,2,2])
        tmp = tf.reshape(tmp,self.net["conv2_2"].shape)
        self.net["pool2_mask"] = tf.cast(tf.greater_equal(self.net["conv2_2"],tmp),dtype=tf.float32)

        self.net["conv3_1"] = self.conv_layer(self.net["pool2"], "conv3_1")
        self.net["conv3_2"] = self.conv_layer(self.net["conv3_1"], "conv3_2")
        self.net["conv3_3"] = self.conv_layer(self.net["conv3_2"], "conv3_3")
        self.net["pool3"] = self.max_pool(self.net["conv3_3"], 'pool3')
        tmp = tf.tile(self.net["pool3"],[1,1,2,2])
        tmp = tf.reshape(tmp,self.net["conv3_3"].shape)
        self.net["pool3_mask"] = tf.cast(tf.greater_equal(self.net["conv3_3"],tmp),dtype=tf.float32)

        self.net["conv4_1"] = self.conv_layer(self.net["pool3"], "conv4_1")
        self.net["conv4_2"] = self.conv_layer(self.net["conv4_1"], "conv4_2")
        self.net["conv4_3"] = self.conv_layer(self.net["conv4_2"], "conv4_3")
        self.net["pool4"] = self.max_pool(self.net["conv4_3"], 'pool4')
        tmp = tf.tile(self.net["pool4"],[1,1,2,2])
        tmp = tf.reshape(tmp,self.net["conv4_3"].shape)
        self.net["pool4_mask"] = tf.cast(tf.greater_equal(self.net["conv4_3"],tmp),dtype=tf.float32)

        self.net["conv5_1"] = self.conv_layer(self.net["pool4"], "conv5_1")
        self.net["conv5_2"] = self.conv_layer(self.net["conv5_1"], "conv5_2")
        self.net["conv5_3"] = self.conv_layer(self.net["conv5_2"], "conv5_3")
        self.net["pool5"] = self.max_pool(self.net["conv5_3"], 'pool5')
        tmp = tf.tile(self.net["pool5"],[1,1,2,2])
        tmp = tf.reshape(tmp,self.net["conv5_3"].shape)
        self.net["pool5_mask"] = tf.cast(tf.greater_equal(self.net["conv5_3"],tmp),dtype=tf.float32)

        self.net["fc6"] = self.fc_layer(self.net["pool5"], "fc6")
        assert self.net["fc6"].get_shape().as_list()[1:] == [4096]
        self.net["relu6"] = tf.nn.relu(self.net["fc6"])

        self.net["fc7"] = self.fc_layer(self.net["relu6"], "fc7")
        self.net["relu7"] = tf.nn.relu(self.net["fc7"])

        self.net["fc8"] = self.fc_layer(self.net["relu7"], "fc8")

        self.net["output"] = tf.nn.softmax(self.net["fc8"], name="prob")

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def build_deconv(self,this_layer="pool5",feature_maps=None):
        layer_index = int(this_layer[4])
        if this_layer.startswith("pool"):
            if layer_index <=2: last_layer = "conv%d_2" % layer_index
            else: last_layer = "conv%d_3" % layer_index
            tmp = tf.tile(feature_maps,[1,1,2,2])
            tmp = tf.reshape(tmp, self.net["%s_mask" % this_layer].shape)
            last_layer_feature_maps = tmp*self.net["%s_mask" % this_layer]
            print("last_layer:%s" % last_layer)
            return self.build_deconv(last_layer,feature_maps=last_layer_feature_maps)
        if this_layer.startswith("conv"):
            num_of_conv_layers = layer_index <= 2 and 2 or 3
            for k in range(num_of_conv_layers,0,-1):
                last_layer = "conv%d_%d" % (layer_index,k)
                print("last_layer:%s" % last_layer)
                relu = tf.nn.relu(feature_maps)
                bias = tf.nn.bias_add(relu,-1*self.get_bias(last_layer))
                output_shape = [1,int(224/self.strides[last_layer]),int(224/self.strides[last_layer]),len(self.data_dict[last_layer][0][0][0])]
                print("output_shape:%s" % str(output_shape))
                last_layer_feature_maps = tf.nn.conv2d_transpose(relu,self.data_dict[last_layer][0],output_shape,strides=[1,1,1,1],padding="SAME")
                feature_maps = last_layer_feature_maps 
            if layer_index == 1:
                return last_layer_feature_maps
            return self.build_deconv("pool%d" % (layer_index-1), feature_maps=last_layer_feature_maps)                
            
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[0]
if __name__ == "__main__":
    deconv_layer = "pool5"
    zoomout = Zoomout_Vgg16("vgg16.npy",deconv_layer=deconv_layer)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img = io.imread("input/test.jpg")
    img = transf.resize(img,(224,224))
    f_ = sess.run(zoomout.net[deconv_layer],feed_dict={zoomout.net["input"]:[img]})
    for i in range(zoomout.channels[deconv_layer]):
        f = np.zeros([1,int(224/zoomout.strides[deconv_layer]),int(224/zoomout.strides[deconv_layer]),zoomout.channels[deconv_layer]])
        #f[:,:,:,i] = f_[:,:,:,i]
        max_9th_value = np.sort(f_[:,:,:,i]).flatten()[-9]
        max_9th_mask = np.greater_equal(f_[:,:,:,i],max_9th_value).astype("int8")
        f[:,:,:,i] = max_9th_mask * f_[:,:,:,i]
        img_v = sess.run(zoomout.net["output_deconv"],feed_dict={zoomout.net["input"]:[img],zoomout.net["input_deconv"]:f})
        mean = np.ones([224,224,3])
        mean[:,:,0] *= VGG_MEAN[2]
        mean[:,:,1] *= VGG_MEAN[1]
        mean[:,:,2] *= VGG_MEAN[0]
        img_v = np.reshape(img_v,[224,224,3])
        img_v += mean
        img_v = img_v.astype("int8")
        io.imsave("output/%s_%d.png" % (deconv_layer,i),img_v)

