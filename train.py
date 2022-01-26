import tensorflow.keras.preprocessing as preprocessing
import tensorflow.keras.utils as utils
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers
import os

train = utils.image_dataset_from_directory(
    'data/train',
    label_mode = 'categorical',
    class_names = None,
    image_size = (300, 300),
    shuffle = True,
    seed = 39,
    validation_split = 0.3,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'data/test',
    label_mode = 'categorical',
    class_names = None,
    image_size = (300, 300),
    shuffle = True,
    seed = 39,
    validation_split = 0.3,
    subset = 'validation',
)

# test = preprocessing.image_dataset_from_directory(
#     'data/test',
#     label_mode = 'categorical',
#     class_names = None,
#     image_size = (300, 300),
# )

class Net():
    def __init__(self, image_size):
        self.model = models.Sequential()
        # input: 300 x 300 x 3
        # convolution
        # frame: 7 x 7
        # strides: 3
        # depth: 8
        self.model.add(layers.Conv2D(8, 7, strides = 3, input_shape = (300, 300, 3), activation = 'relu'))
        # output size: 100 x 100 x 8

        # maxpool
        # frame: 2x2
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # output size: 50 x 50 x 8
        self.model.add(layers.Dropout(0.3))
        
        # convolution
        # frame: 3 x 3
        # strides: 1
        # depth: 16
        self.model.add(layers.Conv2D(16, 3, input_shape = (300, 300, 3), activation = 'relu'))
        # output size: 98 x 98 x 16

        # maxpool
        # frame: 2x2
        self.model.add(layers.MaxPool2D(pool_size=2))
        # output size: 49 x 49 x 16

        self.model.add(layers.Flatten())
        # output size: 38416
        self.model.add(layers.Dense(2048, activation = 'relu'))
        self.model.add(layers.Dense(1024, activation = 'relu'))
        self.model.add(layers.Dense(256, activation = 'relu'))
        self.model.add(layers.Dense(64, activation = 'relu'))
        # values -> possibilities
        self.model.add(layers.Dense(3, activation = 'softmax'))

        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )

    def __str__(self):
        self.model.summary()
        return ""


net = Net((300, 300, 3))
print(net)

checkpoint_path = "./cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# save model every 10 epoch
cp_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose = 1,
    save_weight_only = True,
    save_freq = 'epoch',
    )

net.model.fit(
    train,
    batch_size = 32,
    epochs = 100,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
    callbacks = [cp_callback],
)

net.model.save()