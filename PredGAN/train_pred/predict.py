"""

Predicting and optimization process of PredGAN.

"""

import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Tianyi Zhao"
__credits__ = ["Vinicious L. S. Silva"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Tianyi Zhao"
__email__ = "tianyi.zhao20@imperial.ac.uk"
__status__ = "Development"


def mse_loss(mse, inp, outp, loss_weight, ninput):
    inp = tf.reshape(inp, [-1, ninput])*tf.math.sqrt(loss_weight)
    outp = tf.reshape(outp, [-1, ninput])*tf.math.sqrt(loss_weight)
    return mse(inp, outp)


@tf.function
def opt_step(latent_values, real_coding, loss_weight, mse, generator, optimizer, ntimes, ninput):
    """
    If training for 20 time levels, it finds the loss between the 
    first 20 outputs from the generator and 19 real outputs
    """
    with tf.GradientTape() as tape:
        tape.watch(latent_values)
        gen_output = generator(latent_values, training=False)  #results from generator
        loss = mse_loss(mse, real_coding, gen_output[:,:(ntimes - 1),:,:], loss_weight, ninput)

    gradient = tape.gradient(loss, latent_values)   #gradient of the loss ws to the input
    optimizer.apply_gradients(zip([gradient], [latent_values]))   #applies gradients to the input
    
    return loss


def optimize_coding(real_coding, loss_weight, latent_space, \
             mse, generator, optimizer, ntimes, ninput):
    """
    Returns the optimized input that generates the desired output
    """
    latent_values = tf.random.normal([len(real_coding), latent_space])  
    latent_values = tf.Variable(latent_values)
    
    loss = []
    for epoch in range(5000):
        loss.append(opt_step(latent_values, real_coding, loss_weight, \
            mse, generator, optimizer, ntimes, ninput).numpy())
        
    plt.plot(loss)
    plt.grid()
    plt.show
        
    return latent_values


def optimize_coding_multi(latent_values, real_coding, loss_weight, epochs, \
                         mse, generator, optimizer, ntimes, ninput):
    """
    Returns the optimized input that generates the desired output
    """
    for epoch in range(epochs):
        opt_step(latent_values, real_coding, loss_weight, \
            mse, generator, optimizer, ntimes, ninput)
        
    return latent_values
