
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback

from .utils import *

def simclr_wartmup(encoder,x,batch_size,num_epochs,
                   contrastive_augmentation = None,
                   temperature = 0.1):

    if contrastive_augmentation is None:
         contrastive_augmentation = {'input_shape':x.shape[1:],"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
    
    # with batch sizes selected accordingly
    z_train = np.zeros(x.shape[0])
    unlabeled_dataset_size = z_train.shape[0]
    steps_per_epoch = unlabeled_dataset_size // batch_size
    unlabeled_train_dataset = (tf.data.Dataset.from_tensor_slices((x, z_train))
                                .shuffle(buffer_size=10 * batch_size)
                                .batch(batch_size)
                          )
    # Contrastive pretraining
    pretraining_model = ContrastiveModel(encoder=encoder,
                                         contrastive_augmentation=contrastive_augmentation)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                 initial_learning_rate=1e-3,
                                 decay_steps=50,
                                 decay_rate=0.95)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    pretraining_model.compile(contrastive_optimizer=opt,
                              probe_optimizer=keras.optimizers.Adam(),
                              )
    ## Supervised finetuning of the pretrained encoder
    # We then finetune the encoder on the labeled examples, by attaching
    # a single randomly initalized fully connected classification layer on its top.
    pretraining_history = pretraining_model.fit(unlabeled_train_dataset,
                                                epochs=num_epochs,
                                                verbose=0,
                                                callbacks=[TqdmCallback(verbose=0)]
                                                )
    return pretraining_model.encoder


# Define the contrastive model with model-subclassing
class ContrastiveModel(keras.Model):
    def __init__(self,encoder,contrastive_augmentation,temperature=0.1):
        super().__init__()

        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
#         if encoder is None:
#             self.encoder = get_encoder()
#         else:
        self.encoder = encoder
        width = self.encoder.output_shape[1]
        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(width,)),
                layers.Dense(width, activation="relu"),
                layers.Dense(width),
            ],
            name="projection_head",
        )
        # Single dense layer for linear probing
#         self.linear_probe = keras.Sequential(
#             [layers.InputLayer(input_shape=(width,)), layers.Dense(10)], name="linear_probe"
#         )

        self.encoder.summary()
        self.projection_head.summary()
#         self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
#         self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
#         self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
#             self.probe_loss_tracker,
#             self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        unlabeled_images,_ = data

        images = unlabeled_images
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

#     def train_step(self, data):
#         (unlabeled_images, _), (labeled_images, labels) = data

#         # Both labeled and unlabeled images are used, without labels
#         images = tf.concat((unlabeled_images, labeled_images), axis=0)
#         # Each image is augmented twice, differently
#         augmented_images_1 = self.contrastive_augmenter(images)
#         augmented_images_2 = self.contrastive_augmenter(images)
#         with tf.GradientTape() as tape:
#             features_1 = self.encoder(augmented_images_1)
#             features_2 = self.encoder(augmented_images_2)
#             # The representations are passed through a projection mlp
#             projections_1 = self.projection_head(features_1)
#             projections_2 = self.projection_head(features_2)
#             contrastive_loss = self.contrastive_loss(projections_1, projections_2)
#         gradients = tape.gradient(
#             contrastive_loss,
#             self.encoder.trainable_weights + self.projection_head.trainable_weights,
#         )
#         self.contrastive_optimizer.apply_gradients(
#             zip(
#                 gradients,
#                 self.encoder.trainable_weights + self.projection_head.trainable_weights,
#             )
#         )
#         self.contrastive_loss_tracker.update_state(contrastive_loss)

#         # Labels are only used in evalutation for an on-the-fly logistic regression
#         preprocessed_images = self.classification_augmenter(labeled_images)
#         with tf.GradientTape() as tape:
#             features = self.encoder(preprocessed_images)
#             class_logits = self.linear_probe(features)
#             probe_loss = self.probe_loss(labels, class_logits)
#         gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
#         self.probe_optimizer.apply_gradients(
#             zip(gradients, self.linear_probe.trainable_weights)
#         )
#         self.probe_loss_tracker.update_state(probe_loss)
#         self.probe_accuracy.update_state(labels, class_logits)

#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, data):
#         labeled_images, labels = data

#         # For testing the components are used with a training=False flag
#         preprocessed_images = self.classification_augmenter(
#             labeled_images, training=False
#         )
#         features = self.encoder(preprocessed_images, training=False)
#         class_logits = self.linear_probe(features, training=False)
#         probe_loss = self.probe_loss(labels, class_logits)
#         self.probe_loss_tracker.update_state(probe_loss)
#         self.probe_accuracy.update_state(labels, class_logits)

#         # Only the probe metrics are logged at test time
#         return {m.name: m.result() for m in self.metrics[2:]}


