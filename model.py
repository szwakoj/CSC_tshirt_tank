import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tf.keras import layers, Model
from tf.keras.applications.resnet50 import ResNet50
import cv2

#Tina face architecture (https://arxiv.org/pdf/2011.13183v3.pdf), https://arxiv.org/pdf/1612.03144.pdf , https://arxiv.org/pdf/1708.02002.pdf, as base
#will alter as needed to optimize for our edge tpu architecture

class Feature_Pyramid(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(name='feature_pyramid', **kwargs)
        self.backbone = get_backbone()
        self.conv1x1_first_iter = layers.Conv2D(256, 1, name="first iter")
        self.conv1x1_trans_4 = layers.Conv2D(256, 1, name="transfer_layer_4")
        self.conv1x1_trans_3 = layers.Conv2D(256, 1, name="transfer_layer_3")
        self.conv1x1_trans_2 = layers.Conv2D(256, 1, name="transfer_layer_2")
        self.conv3x3_p4 = layers.Conv2D(256, 3, 1, name="final_conv_5", padding="same")
        self.conv3x3_p4 = layers.Conv2D(256, 3, 1, name="final_conv_4", padding="same")
        self.conv3x3_p3 = layers.Conv2D(256, 3, 1, name="final_conv_3", padding="same")
        self.conv3x3_p2 = layers.Conv2D(256, 3, 1, name="final_conv_2", padding="same")
        self.upsample = layers.UpSampling2D(2, interpolation="nearest", padding="same")

    def get_backbone():
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
        p4 = t4 + self.upsample(p5)
        p4 = self.conv3x3_p4(p4)

        t3 = self.conv1x1_trans_3(c3)
        p3 = t3 + self.upsample(p4)
        p3 = self.conv3x3_p3(p3)

        t2 = self.conv1x1_trans_2(c2)
        p2 = t2 + self.upsample(p3)
        p2 = self.conv3x3_p2(p2)

        return p2, p3, p4, p5

class Face_Model(Model):
    def __init__(sel):
        super().__init__()
        self.model = build_model()

    def build_model(self):
        ###build functional model


        ##Feature extractor

        #import ResNet50
        res_net = ResNet50(weights='imagenet')

        #create Feature Pyramid

        ##Inception Block

        # head

        ##Last Heads

        #Classification

        #Regression

        #IoU Aware

    def compile(self):
        super().compile()

    def train_step(self, gen_data):
        #either custom single train_step or use custom tf.model.fit() to train


