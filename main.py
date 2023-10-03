import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, Model
import cv2


class Crowd_Model(Model):
    def __init__(sel):
        super().__init__()
        self.model = build_model()

    def build_model(self):
        #build functional model

    def compile(self):
        super().compile()

    def train_step(self, gen_data):
        #either custom single train_step or use custom tf.model.fit() to train




