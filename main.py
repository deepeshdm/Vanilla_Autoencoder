
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt

# We'll discard labels since we dont need them.
(x_train,_),(x_test,_) = mnist.load_data()

# Scaling values to [0,1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flattening each 28x28 image to vector of size 784.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


#--------------------------------------------------#

# smaller bottleneck --> more compressed representation
bottleneck_size = 30
flattened_image_size = 28*28

input = Input(shape=(flattened_image_size,))
encoded = Dense(bottleneck_size,activation="relu")(input)
decoded = Dense(flattened_image_size,activation="sigmoid")(encoded)

# Maps input image to decoder's reconstructed image.
# AUTOENCODER
autoencoder = Model(input,decoded)

#--------------------------------------------------#

# creating seperate encoder & decoder models to generate
# images after autoencoder is trained.

# ENCODER
encoder = Model(input,encoded)

# DECODER
encoded_input = Input(shape=(bottleneck_size,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

#--------------------------------------------------#

# training the autoencoder.
autoencoder.compile(optimizer="adam",loss='binary_crossentropy',metrics=["accuracy"])
# using self-supervised approach.
autoencoder.fit(x_train,x_train,epochs=50,batch_size=256,
                shuffle=True,validation_data=(x_test,x_test))

#--------------------------------------------------#

# Reconstructing test images using trained autoencoder.

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#--------------------------------------------------#

n = 10
plt.figure(figsize=(15, 4))

for i in range(n):

    # plotting actual images
    plt.subplot(2,n,i+1)
    plt.gray()
    plt.axis("off")
    plt.imshow(x_test[i].reshape(28,28))

    # plotting reconstructed images
    plt.subplot(2,n,i+1+n)
    plt.gray()
    plt.axis("off")
    plt.imshow(decoded_imgs[i].reshape(28,28))

plt.show()

#--------------------------------------------------#

