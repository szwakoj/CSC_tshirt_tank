import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tf.keras import layers, Model
from tf.keras.applications.resnet50 import ResNet50
import cv2

import utils

#Tina face architecture (https://arxiv.org/pdf/2011.13183v3.pdf), https://arxiv.org/pdf/1612.03144.pdf , https://arxiv.org/pdf/1708.02002.pdf, as base
#will alter as needed to optimize for our edge tpu architecture
class Face_Model(Model):
    def __init__(self, **kwargs):
        super().__init__(name="PolyFace", **kwargs)
        self.fp = FeaturePyramid()
        self.num_anchors = 4
        self.c_head = self.generate_head(self.num_anchors)
        self.b_head = self.generate_head(self.num_anchors * 4)

    def generate_head(self, out_filters):
        if out_filters == self.num_anchors:
            name = "class_head"
        else:
            name = "reg_head"

        input = layers.Input(shape=[None, None, 256])
        kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
        for i in range(4):
            head = layers.Conv2D(256, 3, padding="same", name=name+"_conv3x3_"+str(i), kernel_initializer=kernel_init)(input)
            head = layers.ReLU()(head)

        if name == "classification_head":
            prob = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
            head = layers.Conv2D(out_filters, 3, padding="same", name=name+"_conv1x1", kernel_initializer=kernel_init, bias_initializer=prob)(head)
            output = layers.Activation('sigmoid')(head)
        else:
            head = layers.Conv2D(out_filters, 3, padding="same", name=name+"_conv1x1", kernel_initializer=kernel_init, bias_initializer="zeros")(head)
            output = layers.Activation('linear')(head)

        return Model(input, output)

    def call(self, images, training=False):
        features = self.fp(images, training=training)
        N = tf.shape(images)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.b_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.c_head(feature), [N, -1, 1])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)


class FeaturePyramid(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(name='feature_pyramid', **kwargs)
        self.backbone = self.generate_backbone()
        self.conv1x1_first_iter = layers.Conv2D(256, 1, name="first_iter")
        self.conv1x1_trans_4 = layers.Conv2D(256, 1, name="transfer_layer_4")
        self.conv1x1_trans_3 = layers.Conv2D(256, 1, name="transfer_layer_3")
        self.conv1x1_trans_2 = layers.Conv2D(256, 1, name="transfer_layer_2")
        self.conv3x3_p5 = layers.Conv2D(256, 3, 1, name="final_conv_5", padding="same")
        self.conv3x3_p4 = layers.Conv2D(256, 3, 1, name="final_conv_4", padding="same")
        self.conv3x3_p3 = layers.Conv2D(256, 3, 1, name="final_conv_3", padding="same")
        self.conv3x3_p2 = layers.Conv2D(256, 3, 1, name="final_conv_2", padding="same")
        self.upsample = layers.UpSampling2D(2, interpolation="nearest")
        self.add = layers.Add()

    def generate_backbone(self):
        backbone = ResNet50(include_top=False, input_shape=[None, None, 3], weights="imagenet")
        c2 = backbone.get_layer("conv2_block3_out").output
        c3 = backbone.get_layer("conv3_block4_out").output
        c4 = backbone.get_layer("conv4_block6_out").output
        c5 = backbone.get_layer("conv5_block3_out").output
        return Model(inputs=[backbone.inputs], outputs=[c2, c3, c4, c5])

    def call(self, images, training=False):
        c2, c3, c4, c5 = self.backbone(images, training=training)
        p5 = self.conv1x1_first_iter(c5)
        p5 = self.conv3x3_p5(p5)

        t4 = self.conv1x1_trans_4(c4)
        p4 = self.add([t4, self.upsample(p5)])
        p4 = self.conv3x3_p4(p4)

        t3 = self.conv1x1_trans_3(c3)
        p3 = self.add([t3, self.upsample(p4)])
        p3 = self.conv3x3_p3(p3)

        t2 = self.conv1x1_trans_2(c2)
        p2 = self.add([t2, self.upsample(p3)])
        p2 = self.conv3x3_p2(p2)

        return p2, p3, p4, p5

class FocalLoss(tf.keras.losses.Loss):
    def __init__(**kwargs):

#Anchor box stuff from keras docs
class AnchorBox:
    def __init__(self):
        self.aspects = [1]
        self.scales = [2 ** x/4 for x in range(4)]
        self.num_anchors = len(self.aspects) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32, 64, 128, 256]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspects:
                anchor_h = tf.math.sqrt(area / ratio)
                anchor_w = area / anchor_h
                dims =tf.reshape(
                    tf.stack([anchor_w, anchor_h], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_h, feature_w, level):
        rx = tf.range(feature_w, dtype=tf.float32) + 0.5
        ry = tf.range(feature_h, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_h, feature_w, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, img_h, img_w):
        anchors = [
            self._get_anchors(
                tf.math.ceil(img_h / 2 ** i),
                tf.math.ceil(img_w / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)

