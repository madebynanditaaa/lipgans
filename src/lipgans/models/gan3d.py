import tensorflow as tf
from tensorflow.keras import layers

class VisemeGAN:
    """Simple 3D-GAN wrapper: generator + discriminator + train_step method.
       Generator output range: tanh [-1,1], shape (B, T, H, W, 3).
    """
    def __init__(self, z_dim: int = 100, target_frames: int = 3, img_size=(64,64)):
        self.z_dim = z_dim
        self.target_frames = target_frames
        self.img_size = img_size
        self.gen = self._build_generator()
        self.disc = self._build_discriminator()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def _build_generator(self):
        model = tf.keras.Sequential(name="Generator")
        # Dense -> reshape to (T, 8, 8, C)
        model.add(layers.Dense(self.target_frames * 8 * 8 * 256, input_shape=(self.z_dim,)))
        model.add(layers.Reshape((self.target_frames, 8, 8, 256)))

        model.add(layers.Conv3DTranspose(128, (1,4,4), strides=(1,2,2), padding='same'))
        model.add(layers.BatchNormalization()); model.add(layers.ReLU())

        model.add(layers.Conv3DTranspose(64, (1,4,4), strides=(1,2,2), padding='same'))
        model.add(layers.BatchNormalization()); model.add(layers.ReLU())

        model.add(layers.Conv3DTranspose(3, (1,4,4), strides=(1,2,2), padding='same', activation='tanh'))
        return model

    def _build_discriminator(self):
        T, H, W = self.target_frames, self.img_size[0], self.img_size[1]
        model = tf.keras.Sequential(name="Discriminator")
        model.add(layers.Input(shape=(T, H, W, 3)))
        model.add(layers.Conv3D(64, 4, strides=2, padding='same'))
        model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

        model.add(layers.Conv3D(128, 4, strides=2, padding='same'))
        model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    @tf.function
    def train_step(self, real_clips, g_opt, d_opt):
        """Single training step. real_clips shape: (B, T, H, W, 3)"""
        batch_size = tf.shape(real_clips)[0]
        noise = tf.random.normal([batch_size, self.z_dim])

        # Discriminator step
        with tf.GradientTape() as d_tape:
            fake_clips = self.gen(noise, training=True)
            real_out = self.disc(real_clips, training=True)
            fake_out = self.disc(fake_clips, training=True)
            d_loss = self.bce(tf.ones_like(real_out), real_out) + self.bce(tf.zeros_like(fake_out), fake_out)
        d_grads = d_tape.gradient(d_loss, self.disc.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, self.disc.trainable_variables))

        # Generator step
        with tf.GradientTape() as g_tape:
            fake_clips = self.gen(noise, training=True)
            fake_out = self.disc(fake_clips, training=True)
            g_loss = self.bce(tf.ones_like(fake_out), fake_out)
        g_grads = g_tape.gradient(g_loss, self.gen.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, self.gen.trainable_variables))

        return g_loss, d_loss
