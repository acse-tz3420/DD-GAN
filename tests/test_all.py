import tensorflow as tf
import numpy as np
import sys
sys.path.append(".")

from PredGAN.preprocessing import *
from PredGAN.architectures.generators import *
from PredGAN.architectures.discriminators import *
from PredGAN.train_pred import *


def test_pod_shape():
    """
    Test if the pod coeffs shape correct
    """
    data_dir = './../data/preprocessed/single_domain/'
    pod_coeffs = np.load(data_dir + 'pod_coeffs_1.npy')

    npod = 5
    X_pod = np.transpose(pod_coeffs.reshape(10, 1999)[:npod])

    assert (X_pod.shape == (len(pod_coeffs[:,:,0]), npod))


def test_gan_input_shape():
    """
    Tests if the input dataset shape correct
    """
    data_dir = './../data/preprocessed/single_domain/'
    pod_coeffs = np.load(data_dir + 'pod_coeffs_1.npy')

    npod = 5
    X_pod = np.transpose(pod_coeffs.reshape(10, 1999)[:npod])
    step = 0.25

    X_deriv_1 = calculate_deriv(X_pod, step=step)
    X_deriv_2 = calculate_deriv(X_deriv_1, step=step)
    X_deriv_3 = calculate_deriv(X_deriv_2, step=step)

    X_train = np.concatenate((X_pod[:-3], X_deriv_1[:-2], X_deriv_2[:-1], \
        X_deriv_3), axis=1)

    nderiv = 3
    ninput = npod * (nderiv + 1)

    ntimes = 20 # Consecutive times for the GAN
    level_step = 4 # step between time levels
    nstep = step * level_step

    X_pod_concat = concat_timesteps(X_train, ntimes, level_step)

    assert (X_pod_concat[0, :, :].shape == (ntimes, ninput))


def test_gan_generator_model():
    """
    Test if the generator model can be build correctly
    """
    npod = 5
    nderiv = 3
    ninput = npod * (nderiv + 1)
    ntimes = 20

    generator = make_generator_model_deriv_3()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    assert (generated_image[0, :, :, 0].shape == (0, ntimes, ninput, 0))


def test_gan_discriminator_model():
    """
    Test if the discriminator model can be build correctly
    """
    npod = 5
    nderiv = 3
    ninput = npod * (nderiv + 1)
    ntimes = 20

    generator = make_generator_model_deriv_3()
    discriminator = make_discriminator_model(ntimes=ntimes, ninput=ninput)

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    decision = discriminator(generated_image)

    assert (decision.shape == (1, 1))
