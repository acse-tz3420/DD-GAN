"""

Training process of PredGAN.

"""

import tensorflow as tf
import time
import wandb

__author__ = "Tianyi Zhao"
__credits__ = ["Vinicious L. S. Silva"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Tianyi Zhao"
__email__ = "tianyi.zhao20@imperial.ac.uk"
__status__ = "Development"


def discriminator_loss(real_output, fake_output):
    """
    This method returns a helper function to compute 
    discriminator cross entropy loss
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    """
    This method returns a helper function to compute 
    discriminator cross entropy loss
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(batch, BATCH_SIZE, latent_space, generator_mean_loss, 
               discriminator_mean_loss, generator_optimizer, 
               discriminator_optimizer, generator, discriminator):
    """
    Notice the use of `tf.function`
    This annotation causes the function to be "compiled".
    """
    noise = tf.random.normal([BATCH_SIZE, latent_space])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, 
        generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, 
        discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(
        gradients_of_generator, 
        generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(
        gradients_of_discriminator, 
        discriminator.trainable_variables))

    generator_mean_loss(gen_loss)
    discriminator_mean_loss(disc_loss)


def train(dataset, epochs, BATCH_SIZE, latent_space, generator_mean_loss, 
          discriminator_mean_loss, generator_optimizer, 
          discriminator_optimizer, generator_summary_writer, 
          discriminator_summary_writer, generator, 
          discriminator, gan, notebookName):
    """
    Training the model with this function.
    """
    hist = []
    for epoch in range(epochs):
        start = time.time()
        print("Epoch {}/{}".format(epoch + 1, epochs))   

        for batch in dataset:
            train_step(batch, BATCH_SIZE, latent_space, generator_mean_loss, 
                       discriminator_mean_loss, generator_optimizer, 
                       discriminator_optimizer, generator, discriminator)

        with generator_summary_writer.as_default():
            tf.summary.scalar('loss', generator_mean_loss.result(), step=epoch)

        with discriminator_summary_writer.as_default():
            tf.summary.scalar('loss', discriminator_mean_loss.result(), 
                              step=epoch)

        hist.append([generator_mean_loss.result().numpy(), 
                     discriminator_mean_loss.result().numpy()])

        # wandb.log is used to record some logs (accuracy, loss and epoch), 
        # so that you can check the performance of the network at any time
        wandb.log({
            "loss_G": generator_mean_loss.result().numpy(),
            "loss_D": discriminator_mean_loss.result().numpy()
        })

        generator_mean_loss.reset_states()
        discriminator_mean_loss.reset_states()

        print("discriminator", "loss: {:.6f}".format(hist[-1][1]), end=' - ')
        print("generator", "loss: {:.6f}".format(hist[-1][0]), end=' - ')
        print('{:.0f}s'.format(time.time()-start))

        # Global variables are used below
        if epoch % 1000 == 0: 
    
            # Save model
            gan.save('ganmodels/'+notebookName[:-6]+'-'+str(epoch)+'.h5')    

    return hist
