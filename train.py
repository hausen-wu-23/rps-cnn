import tensorflow.keras.preprocessing as preprocessing
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

train = preprocessing.image_dataset_from_directory(
    'data/train',
    label_mode = 'categorical',
    class_names = None,
    image_size = (300, 300),
    shuffle = True,
    seed = 39,
)

validation = preprocessing.image_dataset_from_directory(
    'data/validation',
    label_mode = 'categorical',
    class_names = None,
    image_size = (300, 300),
    shuffle = True,
    seed = 39,
)

test = preprocessing.image_dataset_from_directory(
    'data/test',
    label_mode = 'categorical',
    class_names = None,
    image_size = (300, 300),
)

class Net():
    def __init__(self, image_size):
        self.model = models.Sequential()
        # input: 300 x 300 x 3
        # convolution
        # frame: 7 x 7
        # strides: 3
        # depth: 8
        self.model.add(layers.Conv2D(8, 7, strides = 3, input_shape = image_size, activation = 'relu'))
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
        self.model.add(layers.Conv2D(16, 3, input_shape = image_size, activation = 'relu'))
        # output size: 98 x 98 x 16

        