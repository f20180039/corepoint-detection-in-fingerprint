import argparse
import os
import sys
from typing import TextIO

import cv2
import matplotlib
import numpy as np
import numpy.random as rng
import scipy
from keras import backend as K
from keras import models
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Flatten, Dropout, merge, Reshape, Conv2D, MaxPooling2D, UpSampling2D, \
    Conv2DTranspose, ZeroPadding2D, Add
from keras.layers.core import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adadelta, RMSprop, SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

'''Set Keras image format '''
K.set_image_data_format('channels_last')

x_shape = 256
y_shape = 320
channels = 1

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--epochs', dest='epochs', default=100, help='no of epochs to train')
parser.add_argument('--batch_size', dest='batch_size', default=8, help='batch size')
args = parser.parse_args()

if args.phase == 'train':
    train_folder = input("Enter training data folder: ")
    model_folder = input("Enter folder to save model: ")
    train_img = os.path.join(train_folder, "data_used")
    train_gt = os.path.join(train_folder, "g_Truth")
    mask_gt = os.path.join(train_folder, "Mask_gt")
    val_sample = os.path.join(model_folder, "Val_sample")
    plot = os.path.join(model_folder, "Plot")
    model_weights = os.path.join(model_folder, "Weights")
    loss_files = os.path.join(model_folder, "Loss_files")
else:
    test_input = input("Enter testing data folder: ")
    model_folder = input("Enter saved model folder: ")
    test_folder = input("Enter folder to save test output: ")
    test_img = os.path.join(test_input, "data_used")
    test_gt = os.path.join(test_input, "g_Truth")
    model_weights = os.path.join(model_folder, "Weights")
    test_visual = os.path.join(test_folder, "Visual_predictions")
    mask_visual = os.path.join(test_folder, "mask_predictions")


###########################################  Load Data  ####################################################
def load_data():
    imagePath = train_img
    gtPath = train_gt
    maskPath = mask_gt
    imageExt = ".bmp"
    maskExt = ".bmp"
    gtExt = "_gt.txt"

    files = []
    files = os.listdir(imagePath)

    images = []
    mask = []
    gt = []
    for file in files:
        filename = file.split('.')[0]
        imagefile = os.path.join(imagePath, file)
        maskfile = os.path.join(maskPath, filename + maskExt)
        gtfile = os.path.join(gtPath, filename + gtExt)
        if not (os.path.exists(imagefile)) or not (os.path.exists(maskfile)) or not (os.path.exists(gtfile)):
            continue

        im = cv2.imread(imagefile, 0)
        original_shape1, original_shape2 = im.shape
        im = cv2.resize(im, (x_shape, y_shape))
        im = im[:, :, np.newaxis]
        images.append(im)

        im = cv2.imread(maskfile, 0)
        original_shape1, original_shape2 = im.shape
        im = cv2.resize(im, (x_shape, y_shape))
        im = im[:, :, np.newaxis]
        mask.append(im)

        f = open(gtfile, 'r')
        y, x = map(float, f.readline().split())
        x = (x * x_shape) / original_shape2
        y = (y * y_shape) / original_shape1
        gt.append((x, y))

    x = np.array(images)
    y = np.array(gt)
    z = np.array(mask)
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(x, y, z, test_size=0.2)
    return X_train, X_test, Y_train, Y_test, Z_train, Z_test


###########################################  Encoder  ####################################################
def Encoder(input_img):
    Econv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name="block1_conv1")(input_img)
    Econv1_1 = BatchNormalization()(Econv1_1)
    Econv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name="block1_conv2")(Econv1_1)
    Econv1_2 = BatchNormalization()(Econv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="block1_pool1")(Econv1_2)

    Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv1")(pool1)
    Econv2_1 = BatchNormalization()(Econv2_1)
    Econv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv2")(Econv2_1)
    Econv2_2 = BatchNormalization()(Econv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="block2_pool1")(Econv2_2)

    Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name="block3_conv1")(pool2)
    Econv3_1 = BatchNormalization()(Econv3_1)
    Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="block3_conv2")(Econv3_1)
    Econv3_2 = BatchNormalization()(Econv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="block3_pool1")(Econv3_2)

    encoded = Model(inputs=input_img, outputs=[pool3, Econv1_2, Econv2_2, Econv3_2])

    return encoded


#########################################  Bottleneck ##################################################
def neck(input_layer):
    Nconv = Conv2D(256, (3, 3), padding="same", name="neck1")(input_layer)
    Nconv = BatchNormalization()(Nconv)
    Nconv = Conv2D(128, (3, 3), padding="same", name="neck2")(Nconv)
    Nconv = BatchNormalization()(Nconv)

    neck_model = Model(input_layer, Nconv)
    return neck_model


#########################################  Hourglass ##################################################
def Hourglass(input_layer):
    conv_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_conv1")(input_layer)

    conv1_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_block1_conv1")(conv_1)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_block1_conv2")(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_block1_conv3")(conv1_2)
    conv1_3 = BatchNormalization()(conv1_3)
    residual1 = Add(name="hg_block1_add")([conv_1, conv1_3])

    pool1_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="hg_block1_pool1")(residual1)  # 56

    branch1_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_branch_block1_conv1")(residual1)
    branch1_1 = BatchNormalization()(branch1_1)
    branch1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_branch_block1_conv2")(branch1_1)
    branch1_2 = BatchNormalization()(branch1_2)
    branch1_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_branch_block1_conv3")(branch1_2)
    branch1_3 = BatchNormalization()(branch1_3)
    bresidual1 = Add(name="hg_branch_block1_add")([residual1, branch1_3])

    conv2_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_block2_conv1")(pool1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_block2_conv2")(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_block2_conv3")(conv2_2)
    conv2_3 = BatchNormalization()(conv2_3)
    residual2 = Add(name="hg_block2_add")([pool1_1, conv2_3])

    pool2_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="hg_block2_pool1")(residual2)  # 28

    branch2_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_branch_block2_conv1")(residual2)
    branch2_1 = BatchNormalization()(branch2_1)
    branch2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_branch_block2_conv2")(branch2_1)
    branch2_2 = BatchNormalization()(branch2_2)
    branch2_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_branch_block2_conv3")(branch2_2)
    branch2_3 = BatchNormalization()(branch2_3)
    bresidual2 = Add(name="hg_branch_block2_add")([residual2, branch2_3])

    conv3_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_block3_conv1")(pool2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_block3_conv2")(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_block3_conv3")(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    residual3 = Add(name="hg_block3_add")([pool2_1, conv3_3])

    pool3_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="hg_block3_pool1")(residual3)  # 14

    branch3_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_branch_block3_conv1")(residual3)
    branch3_1 = BatchNormalization()(branch3_1)
    branch3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_branch_block3_conv2")(branch3_1)
    branch3_2 = BatchNormalization()(branch3_2)
    branch3_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_branch_block3_conv3")(branch3_2)
    branch3_3 = BatchNormalization()(branch3_3)
    bresidual3 = Add(name="hg_branch_block3_add")([residual3, branch3_3])

    ###########################BOTLLENECK######################################

    conv4_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_block4_conv1")(pool3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_block4_conv2")(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_block4_conv3")(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    residual4 = Add(name="hg_block4_add")([pool3_1, conv4_3])

    conv5_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_block5_conv1")(residual4)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_block5_conv2")(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_block5_conv3")(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    residual5 = Add(name="hg_block5_add")([residual4, conv5_3])

    #############################################################################

    up1_1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same', name="hg_up1")(residual5)
    up1_1 = BatchNormalization()(up1_1)  # 28
    add1 = Add(name="hg_up1_add")([up1_1, bresidual3])

    uconv1_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_upconv1_1")(add1)
    uconv1_1 = BatchNormalization()(uconv1_1)
    uconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_upconv1_2")(uconv1_1)
    uconv1_2 = BatchNormalization()(uconv1_2)
    uconv1_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_upconv1_3")(uconv1_2)
    uconv1_3 = BatchNormalization()(uconv1_3)
    uresidual1 = Add(name="hg_upblock1_add")([add1, uconv1_3])

    up2_1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same', name="hg_up2")(uresidual1)
    up2_1 = BatchNormalization()(up2_1)  # 56
    add2 = Add()([up2_1, bresidual2])

    uconv2_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_upconv2_1")(add2)
    uconv2_1 = BatchNormalization()(uconv2_1)
    uconv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_upconv2_2")(uconv2_1)
    uconv2_2 = BatchNormalization()(uconv2_2)
    uconv2_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_upconv2_3")(uconv2_2)
    uconv2_3 = BatchNormalization()(uconv2_3)
    uresidual2 = Add(name="hg_upblock2")([add2, uconv2_3])

    up3_1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same', name="hg_up3")(uresidual2)
    up3_1 = BatchNormalization()(up3_1)  # 112
    add3 = Add()([up3_1, bresidual1])

    uconv3_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_upconv3_1")(add3)
    uconv3_1 = BatchNormalization()(uconv3_1)
    uconv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="hg_upconv3_2")(uconv3_1)
    uconv3_2 = BatchNormalization()(uconv3_2)
    uconv3_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name="hg_upconv3_3")(uconv3_2)
    uconv3_3 = BatchNormalization()(uconv3_3)
    uresidual3 = Add()([add3, uconv3_3])

    out_hg = Conv2D(128, (1, 1), activation='relu', padding='same', name="hg_out")(uresidual3)
    Hg = Model(input_layer, out_hg)

    return Hg


##########################################  Decoder   ##################################################
def Decoder(inp):
    up1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name="upsample_1")(inp[0])
    up1 = BatchNormalization()(up1)
    up1 = tf.concat([up1, inp[3]], axis=3, name="merge_1")  # testing
    Upconv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name="Upconv1_1")(up1)
    Upconv1_1 = BatchNormalization()(Upconv1_1)
    Upconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="Upconv1_2")(Upconv1_1)
    Upconv1_2 = BatchNormalization()(Upconv1_2)

    up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name="upsample_2")(Upconv1_2)
    up2 = BatchNormalization()(up2)
    up2 = tf.concat([up2, inp[2]], axis=3, name="merge_2")  # testing
    Upconv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name="Upconv2_1")(up2)
    Upconv2_1 = BatchNormalization()(Upconv2_1)
    Upconv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name="Upconv2_2")(Upconv2_1)
    Upconv2_2 = BatchNormalization()(Upconv2_2)

    up3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='relu', padding='same', name="upsample_3")(Upconv2_2)
    up3 = BatchNormalization()(up3)
    up3 = tf.concat([up3, inp[1]], axis=3, name="merge_3")  # testing
    Upconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name="Upconv3_1")(up3)
    Upconv3_1 = BatchNormalization()(Upconv3_1)
    Upconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name="Upconv3_2")(Upconv3_1)
    Upconv3_2 = BatchNormalization()(Upconv3_2)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name="decoded")(Upconv3_2)
    convnet = Model(inputs=inp, outputs=decoded, name="Mask_output")
    return convnet


###########################################  Regressor  ####################################################
def Regressor(input_img, decoded):
    merg1 = tf.concat([input_img, decoded], axis=3, name="merge_r1")  # testing
    reg_conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name="block1_conv1")(merg1)
    reg_conv1_1 = BatchNormalization()(reg_conv1_1)
    reg_conv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name="block1_conv2")(reg_conv1_1)
    reg_conv1_2 = BatchNormalization()(reg_conv1_2)
    reg_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="block1_pool1")(reg_conv1_2)

    reg_conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv1")(reg_pool1)
    reg_conv2_1 = BatchNormalization()(reg_conv2_1)
    reg_conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv2")(reg_conv2_1)
    reg_conv2_2 = BatchNormalization()(reg_conv2_2)
    reg_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="block2_pool1")(reg_conv2_2)

    reg_conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name="block3_conv1")(reg_pool2)
    reg_conv3_1 = BatchNormalization()(reg_conv3_1)
    reg_conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="block3_conv2")(reg_conv3_1)
    reg_conv3_2 = BatchNormalization()(reg_conv3_2)
    reg_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="block3_pool1")(reg_conv3_2)

    reg_flat = Flatten()(reg_pool3)
    fc1 = Dense(256, activation='relu')(reg_flat)
    fc2 = Dense(64, activation='relu')(fc1)
    fc3 = Dense(16, activation='relu')(fc2)
    fc4 = Dense(2, activation='relu')(fc3)
    regress = Model(inputs=[input_img, decoded], outputs=fc4, name="Output_layer")
    return regress


#########################################################################################################

##########################################'''Initialise the model.'''####################################

# Encoder
input_img = Input(shape=(y_shape, x_shape, channels))
encoded = Encoder(input_img)
# encoded.load_weights(os.path.join(model_weights,'encoded.h5')) #testing
# encoded.trainable = False #testing

# Decoder
HG_ = Input(shape=(y_shape // (2 ** 3), x_shape // (2 ** 3), 128))
conv1_l = Input(shape=(y_shape, x_shape, 16))
conv2_l = Input(shape=(y_shape // (2 ** 1), x_shape // (2 ** 1), 64))
conv3_l = Input(shape=(y_shape // (2 ** 2), x_shape // (2 ** 2), 128))
decoded = Decoder([HG_, conv1_l, conv2_l, conv3_l])
# decoded.load_weights(os.path.join(model_weights,'decoded.h5')) #testing
# decoded.trainable = False #testing

# #BottleNeck
# Neck_input = Input(shape = (y_shape//(2**3), x_shape//(2**3),128))
# neck = neck(Neck_input)
# # neck.load_weights('UNet.h5', by_name = True)
# neck.trainable = True

# Hourglass_1
HG_input = Input(shape=(y_shape // (2 ** 3), x_shape // (2 ** 3), 128))
Hg_1 = Hourglass(HG_input)
Hg_2 = Hourglass(HG_input)
Hg_3 = Hourglass(HG_input)
Hg_4 = Hourglass(HG_input)
# Hg_1.trainable = False #testing
# Hg_2.trainable = False #testing
# Hg_3.trainable = False #testing
# Hg_1.load_weights(os.path.join(model_weights,"Hg_1.h5"), by_name = True)
# Hg_1.load_weights(os.path.join(model_weights,"Hg_1.h5")) #testing
# Hg_2.load_weights(os.path.join(model_weights,"Hg_2.h5")) #testing
# Hg_3.load_weights(os.path.join(model_weights,"Hg_3.h5")) #testing

ae_output = decoded(
    [Hg_4(Hg_3(Hg_2(Hg_1(encoded(input_img)[0])))), encoded(input_img)[1], encoded(input_img)[2], encoded(input_img)[3]])

# Regressor
mask_ae = Input(shape=(y_shape, x_shape, channels))
reg = Regressor(input_img, mask_ae)
# reg.load_weights(os.path.join(model_weights,'Regressor.h5')) #testing
# reg.trainable = False #testing

# Combined
output_img = reg([input_img, ae_output])
model = Model(inputs=input_img, outputs=[output_img, ae_output])

if os.path.exists(os.path.join(model_weights,'Finger_AE_Hourglass_Regressor.h5')):
    model.load_weights(os.path.join(model_weights,'Finger_AE_Hourglass_Regressor.h5'), by_name = True)

model.summary()
losses = {
    "Mask_output": "binary_crossentropy",
    "Output_layer": "mean_squared_error"
}
model.compile(optimizer=Adam(0.0005), loss=losses)

#########################################################################################################

if args.phase == 'train':
    gtPath = train_gt
    train_files = os.listdir(train_img)
    print("Data_splitting..")
    X_train, X_test, Y_train, Y_test, Mask_train, Mask_test = load_data()
    X_train = np.asarray(X_train, np.float16) / 255
    X_test = np.asarray(X_test, np.float16) / 255
    Mask_train = np.asarray(Mask_train, np.float16) / 255
    Mask_test = np.asarray(Mask_test, np.float16) / 255

    saveModel = os.path.join(model_weights, 'Finger_AE_Hourglass_Regressor.h5')
    numEpochs = args.epochs
    batch_size = args.batch_size
    num_batches = int(len(X_train) / batch_size)
    print("Number of batches: %d\n" % num_batches)
    loss = []
    val_loss = []
    acc = []
    val_acc = []
    epoch = 0
    r_c = 0

    while epoch < numEpochs:
        history = model.fit(X_train, {'Mask_output': Mask_train, 'Output_layer': Y_train}, batch_size=batch_size,
                            epochs=1, validation_data=(X_test, {'Mask_output': Mask_test, 'Output_layer': Y_test}),
                            shuffle=True, verbose=1)
        model.save_weights(saveModel)

        epoch = epoch + 1
        print("EPOCH NO. : " + str(epoch) + "\n")
        loss.append(float(history.history['loss'][0]))
        val_loss.append(float(history.history['val_loss'][0]))
        loss_arr = np.asarray(loss)
        e = range(epoch)
        plt.plot(e, loss_arr)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Training Loss')
        plt.savefig(os.path.join(plot, str(epoch) + '.png'))
        plt.close()

        loss1 = np.asarray(loss)
        val_loss1 = np.asarray(val_loss)
        acc1 = np.asarray(acc)
        val_acc1 = np.asarray(val_acc)
        np.savetxt(os.path.join(loss_files, 'Loss.txt'), loss1)
        np.savetxt(os.path.join(loss_files, 'Val_Loss.txt'), val_loss1)
        np.savetxt(os.path.join(loss_files, 'Acc.txt'), acc1)
        np.savetxt(os.path.join(loss_files, 'Val_Acc.txt'), val_acc1)

        s = rng.randint(len(train_files))
        filename = train_files[s]
        path = os.path.join(train_img, filename)
        save_path = os.path.join(val_sample, filename)
        x_test = cv2.imread(path, 0)
        x_test = cv2.resize(x_test, (x_shape, y_shape))
        x_test = x_test[:, :, np.newaxis]
        x_test = np.array([x_test])
        x_test = np.asarray(x_test, np.float16) / 255
        y_test = model.predict(x_test)

        x_test = cv2.imread(path, 0)
        original_shape1, original_shape2 = x_test.shape
        x_test = cv2.resize(x_test, (x_shape, y_shape))
        x_test = x_test[:, :, np.newaxis]
        x_test = np.array(x_test)

        name = filename.split('.')[0]
        gtfile = os.path.join(gtPath, name + "_gt.txt")
        if os.path.exists(gtfile) is False:
            continue
        f = open(gtfile, 'r')
        y, x = map(float, f.readline().split())
        x = (x * x_shape) / original_shape2
        y = (y * y_shape) / original_shape1

        cv2.circle(x_test, (int(y_test[0][0][0]), int(y_test[0][0][1])), 4, (0, 0, 255), -1)
        cv2.circle(x_test, (int(x), int(y)), 4, (255, 0, 0), -1)
        print("point",y_test[0][0][0],y_test[0][0][1])
        cv2.imwrite(save_path, x_test)

else:
    if os.path.exists(test_img):
        files = os.listdir(test_img)
    else:
        sys.exit("Invalid Path")

    i = 0
    for filename in files:
        i += 1
        print(i)
        path = os.path.join(test_img, filename)
        save_path = os.path.join(test_visual, filename)
        x_test = cv2.imread(path, 0)
        x_test = cv2.resize(x_test, (x_shape, y_shape))
        x_test = x_test[:, :, np.newaxis]
        x_test = np.array([x_test])
        x_test = np.asarray(x_test, np.float16) / 255
        y_test = model.predict(x_test)

        x_test = cv2.imread(path, 0)
        original_shape1, original_shape2 = x_test.shape
        x_test = cv2.resize(x_test, (x_shape, y_shape))
        x_test = x_test[:, :, np.newaxis]
        x_test = np.array(x_test)

        name = filename.split('.')[0]
        gtfile = os.path.join(test_gt, name + "_gt.txt")
        if (os.path.exists(gtfile)):
            f: TextIO = open(gtfile, 'r')
        else:
            continue
        try:
            y, x = map(float, f.readline().split())
        except:
            print(filename, " has no core point")
        x = (x * x_shape) / original_shape2
        y = (y * y_shape) / original_shape1

        # print(int(y_test[0][0][0]),int(y_test[0][0][1]))
        cv2.circle(x_test, (int(y_test[0][0][0]), int(y_test[0][0][1])), 4, (0, 0, 255), -1)  # black
        cv2.circle(x_test, (int(x), int(y)), 4, (255, 0, 0), -1)  # white
        ############################################# To save coordinates in files ########################
        f = open(os.path.join(os.path.join(test_folder, "Predictions"), name + "_gt.txt"), 'w+')
        f.write(str((y_test[0][0][1] * original_shape1) / y_shape) + " " + str(
            (y_test[0][0][0] * original_shape2) / x_shape))
        f.close()

        # print(y_test)
        mask_pred = np.array(y_test[1][0]) * 255
        mask_save_path = os.path.join(mask_visual, filename)
        cv2.imwrite(mask_save_path, mask_pred)
        ###################################################################################################
        cv2.imwrite(save_path, x_test) #D:/Core_point_detection/Core_point_GT/fvc2002_db1
