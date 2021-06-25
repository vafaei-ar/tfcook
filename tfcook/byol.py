

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from .utils import *

# 512 (h) -> 256 -> 128 (z)
class ProjectionHead(tf.keras.Model):

    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=256)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(units=128)

    def call(self, inp, training=False):
        x = self.fc1(inp)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x

def byol_loss(p, z):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)

    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(similarities)


# Load CIFAR-10 dataset
# data = CIFAR10()
def byol_warmup(encoder,x,batch_size,num_epochs,
                contrastive_augmentation = None
               ):

    from tqdm.notebook import tqdm
    if contrastive_augmentation is None:
         contrastive_augmentation = {'input_shape':x.shape[1:],"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}

    # Instantiate networks
    # f_online = get_encoder(input_shape=(image_size, image_size, image_channels),nfilter=PROJECT_DIM)
    target_enc = keras.models.clone_model(encoder)
    f_online = encoder
    g_online = ProjectionHead()
    q_online = ProjectionHead()

    # f_target = get_encoder(input_shape=(image_size, image_size, image_channels),nfilter=PROJECT_DIM)
    f_target = target_enc
    g_target = ProjectionHead()

    # Initialize the weights of the networks
    # (256, 32, 32, 3)
    xtf = tf.random.normal((64,*x.shape[1:]))
    h = f_online(xtf, training=False)
#     print('Initializing online networks...')
#     print('Shape of h:', h.shape)
    z = g_online(h, training=False)
#     print('Shape of z:', z.shape)
    p = q_online(z, training=False)
#     print('Shape of p:', p.shape)

    h = f_target(xtf, training=False)
#     print('Initializing target networks...')
#     print('Shape of h:', h.shape)
    z = g_target(h, training=False)
#     print('Shape of z:', z.shape)

    num_params_f = tf.reduce_sum([tf.reduce_prod(var.shape) for var in f_online.trainable_variables])    
#     print('The encoders have {} trainable parameters each.'.format(num_params_f))


    # Define optimizer
    lr = 1e-3 * batch_size / 512
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
#     print('Using Adam optimizer with learning rate {}.'.format(lr))


    @tf.function
    def train_step_pretraining(x1, x2):  # (bs, 32, 32, 3), (bs, 32, 32, 3)

        # Forward pass
        h_target_1 = f_target(x1, training=True)
        z_target_1 = g_target(h_target_1, training=True)

        h_target_2 = f_target(x2, training=True)
        z_target_2 = g_target(h_target_2, training=True)

        with tf.GradientTape(persistent=True) as tape:
            h_online_1 = f_online(x1, training=True)
            z_online_1 = g_online(h_online_1, training=True)
            p_online_1 = q_online(z_online_1, training=True)

            h_online_2 = f_online(x2, training=True)
            z_online_2 = g_online(h_online_2, training=True)
            p_online_2 = q_online(z_online_2, training=True)

            p_online = tf.concat([p_online_1, p_online_2], axis=0)
            z_target = tf.concat([z_target_2, z_target_1], axis=0)
            loss = byol_loss(p_online, z_target)

        # Backward pass (update online networks)
        grads = tape.gradient(loss, f_online.trainable_variables)
        opt.apply_gradients(zip(grads, f_online.trainable_variables))
        grads = tape.gradient(loss, g_online.trainable_variables)
        opt.apply_gradients(zip(grads, g_online.trainable_variables))
        grads = tape.gradient(loss, q_online.trainable_variables)
        opt.apply_gradients(zip(grads, q_online.trainable_variables))
        del tape

        return loss

    n_train = x.shape[0]
    batches_per_epoch = n_train // batch_size
    losses = []

    pbar = tqdm(total=num_epochs*batches_per_epoch)

    for epoch_id in range(num_epochs):
        for batch_id in range(batches_per_epoch):
            pbar.update(1)
            inds = np.random.randint(0,n_train,batch_size)
            x1, x2 = x[inds],x[inds]


            x1 = get_augmenter(**contrastive_augmentation)(x1)
            x2 = get_augmenter(**contrastive_augmentation)(x2)

            loss = train_step_pretraining(x1, x2)
            losses.append(float(loss))

            # Update target networks (exponential moving average of online networks)
            beta = 0.99

            f_target_weights = f_target.get_weights()
            f_online_weights = f_online.get_weights()
            for i in range(len(f_online_weights)):
                f_target_weights[i] = beta * f_target_weights[i] + (1 - beta) * f_online_weights[i]
            f_target.set_weights(f_target_weights)

            g_target_weights = g_target.get_weights()
            g_online_weights = g_online.get_weights()
            for i in range(len(g_online_weights)):
                g_target_weights[i] = beta * g_target_weights[i] + (1 - beta) * g_online_weights[i]
            g_target.set_weights(g_target_weights)

    return f_online
