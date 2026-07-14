
import tensorflow as tf
from tensorflow.keras import models, layers


class VGG16Model:
    """
    Contains methods for building, training, and evaluating a CNN model using VGG16 architecture. 
    """
    def __init__(self):
        self.model = None
        self.history = None
    
    def build_model(self, learning_rate, dropout):
        base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        # Unfreeze last 30 layers
        base_model.trainable = True
        for layer in base_model.layers[:-5]:
            layer.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid")
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

    def train_model(self, train_data, val_data, epochs=20, callbacks=None):
        self.history = self.model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks)
        self.model.save("models/vgg16.keras")

    def evaluate_model(self, test_data):
        return self.model.evaluate(test_data)