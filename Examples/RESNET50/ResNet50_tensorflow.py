import tensorflow as tf

# Set random seed
tf.random.set_seed(42)  # You can replace 42 with any integer value you prefer

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, in_channels, intermediate_channels, expansion, is_Bottleneck, stride):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_Bottleneck = is_Bottleneck
        self.stride = stride

        if self.in_channels == self.intermediate_channels * self.expansion:
            self.identity = True
        else:
            self.identity = False
            self.projection = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.intermediate_channels * self.expansion, kernel_size=1, strides=stride, padding='valid', use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

        self.relu = tf.keras.layers.ReLU()

        if self.is_Bottleneck:
            self.conv1_1x1 = tf.keras.layers.Conv2D(self.intermediate_channels, kernel_size=1, strides=1, padding='valid', use_bias=False)
            self.batchnorm1 = tf.keras.layers.BatchNormalization()

            self.conv2_3x3 = tf.keras.layers.Conv2D(self.intermediate_channels, kernel_size=3, strides=stride, padding='same', use_bias=False)
            self.batchnorm2 = tf.keras.layers.BatchNormalization()

            self.conv3_1x1 = tf.keras.layers.Conv2D(self.intermediate_channels * self.expansion, kernel_size=1, strides=1, padding='valid', use_bias=False)
            self.batchnorm3 = tf.keras.layers.BatchNormalization()

        else:
            self.conv1_3x3 = tf.keras.layers.Conv2D(self.intermediate_channels, kernel_size=3, strides=stride, padding='same', use_bias=False)
            self.batchnorm1 = tf.keras.layers.BatchNormalization()

            self.conv2_3x3 = tf.keras.layers.Conv2D(self.intermediate_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
            self.batchnorm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        in_x = inputs

        if self.is_Bottleneck:
            x = self.relu(self.batchnorm1(self.conv1_1x1(inputs)))
            x = self.relu(self.batchnorm2(self.conv2_3x3(x)))
            x = self.batchnorm3(self.conv3_1x1(x))

        else:
            x = self.relu(self.batchnorm1(self.conv1_3x3(inputs)))
            x = self.batchnorm2(self.conv2_3x3(x))

        if self.identity:
            x += in_x
        else:
            x += self.projection(in_x)

        x = self.relu(x)

        return x


class ResNet(tf.keras.Model):
    def __init__(self, resnet_variant=[[64, 128, 256, 512], [3, 4, 6, 3], 4, True], in_channels=3, num_classes=200):
        super(ResNet, self).__init__()
        self.channels_list = resnet_variant[0]
        self.repeatition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.block1 = self._make_blocks(64, self.channels_list[0], self.repeatition_list[0], self.expansion, self.is_Bottleneck, stride=1)
        self.block2 = self._make_blocks(self.channels_list[0] * self.expansion, self.channels_list[1], self.repeatition_list[1], self.expansion, self.is_Bottleneck, stride=2)
        self.block3 = self._make_blocks(self.channels_list[1] * self.expansion, self.channels_list[2], self.repeatition_list[2], self.expansion, self.is_Bottleneck, stride=2)
        self.block4 = self._make_blocks(self.channels_list[2] * self.expansion, self.channels_list[3], self.repeatition_list[3], self.expansion, self.is_Bottleneck, stride=2)

        self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.relu(self.batchnorm1(self.conv1(inputs)))
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.average_pool(x)
        x = self.fc1(x)
        return x

    def _make_blocks(self, in_channels, intermediate_channels, num_repeat, expansion, is_Bottleneck, stride):
        layers = []
        layers.append(Bottleneck(in_channels, intermediate_channels, expansion, is_Bottleneck, stride=stride))
        for num in range(1, num_repeat):
            layers.append(Bottleneck(intermediate_channels * expansion, intermediate_channels, expansion, is_Bottleneck, stride=1))
        return tf.keras.Sequential(layers)