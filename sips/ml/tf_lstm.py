import tensorflow as tf


class TfLSTM(tf.keras.Model):
    '''
    subclass model type
    '''
    def __init__(self):
        super(TfLSTM, self).__init__()
        self.l1 = LSTM(100, activation='relu')
        self.l2 = Dense(128, activation='relu')
        self.l3 = Dense(19, activation='softmax')

    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


# def make_model():
# # sequential model

#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(28, 28)),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(10, activation='softmax')
#     ])
#     return model


model = TfLSTM()

with tf.GradientTape() as tape:
  logits = model(images)
  loss_value = loss(logits, labels)
grads = tape.gradient(loss_value, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.traina
