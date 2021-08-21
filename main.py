import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_array_ops import parallel_concat
import tensorflow_addons as tf_addon

num_classes = 100
input_shape = (32,32,3)

#load CIFAR 100 dataset
(x_train,y_train),(x_test,y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape : {x_train.shape} - y_train shape{y_train.shape}")
print(f"x_test shape : {x_test.shape} - y_test shape {y_test.shape}")

from constants import *

# using data augmentation
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(img_size,img_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2,width_factor=0.2),
    ],
    name = "data_augmentation"
)

#compute the mean and variance of the training data for normalization
data_augmentation.layers[0].adapt(x_train)

#multi layer perceptron => feed forward network
def multi_layer_perceptron(x,hidden_units,dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units,activation=tf.nn.gelu)(x)
        x - layers.Dropout(dropout_rate)(x)
    return x

#patch creation
class Patches(layers.layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self,images):
        batch_size = tf.shape(images[0])
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1,self.patch_size,self.patch_size,1],
            strides=[1,self.patch_size,self.patch_size,1],
            rate = [1,1,1,1],
            padding = "VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size,-1,patch_dims])

        return patches

class PatchEncoder(layers.Layer):
    def __init__(self,num_patches,projection_dim):
        super(PatchEncoder,self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches,output_dim=projection_dim)

    def forward(self,patch):
        positions = tf.range(start=0,limit=self.num_patches,data = 1)
        encoded = self.projection(patch) + self.position_embedding

        return encoded


# transformer model
def vit_classifier():
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches,projection_dim)(patches)

    #multiple layers of transformer block
    for _ in range(transformer_layers):
        #layer normalization
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        #create a multi head attention layer
        attention_output = layers.MultiHeadAttention(num_heads=num_heads,key_dim=projection_dim,dropout=0.1)(x1,x1)

        x2 = layers.Add()([attention_output,encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = multi_layer_perceptron(x3,hidden_units=transformer_units)
        encoded_patches = layers.Add()([x3,x2])


    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = multi_layer_perceptron(representation, hidden_units = multi_layer_perceptron, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)

    model = keras.Model(inputs = inputs, outputs = logits)
    return model


def run_experiment(model):
    optimizer = tf_addon.optimizer.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True
    )

    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = batch_size,
        epochs = num_epochs,
        validation_split=0.1,
        callbacks = [checkpoint_callback],
    )


    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test,y_test)
    print(f"Test accuracy : {round(accuracy*100,2)} %")
    print(f"Test top 5 accuracy : {round(top_5_accuracy*100,2)} %")

    return history


transformer = vit_classifier()
history = run_experiment(vit_classifier)
