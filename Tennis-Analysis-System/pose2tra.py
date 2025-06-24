import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization,
    TimeDistributed, Add, Concatenate, MultiHeadAttention
)
from tensorflow.keras.models import Model

# Simple Time2Vector embedding layer (learns a time embedding per time step)
class Time2Vector(tf.keras.layers.Layer):
    def __init__(self, seq_len):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(
            shape=(self.seq_len,), initializer='uniform', trainable=True, name='weights_linear')
        self.bias_linear = self.add_weight(
            shape=(self.seq_len,), initializer='uniform', trainable=True, name='bias_linear')
        self.weights_periodic = self.add_weight(
            shape=(self.seq_len,), initializer='uniform', trainable=True, name='weights_periodic')
        self.bias_periodic = self.add_weight(
            shape=(self.seq_len,), initializer='uniform', trainable=True, name='bias_periodic')

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, num_features)
        time_linear = self.weights_linear * tf.range(self.seq_len, dtype=tf.float32) + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=0)  # shape (1, seq_len)
        time_linear = tf.expand_dims(time_linear, axis=-1) # shape (1, seq_len, 1)

        time_periodic = tf.sin(self.weights_periodic * tf.range(self.seq_len, dtype=tf.float32) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=0)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)

        time_embedding = tf.concat([time_linear, time_periodic], axis=-1)
        return tf.tile(time_embedding, [tf.shape(inputs)[0], 1, 1])  # match batch size

def create_model(src, num_features, ff_dim=512):
    # Encoder
    encoder_inputs = Input(shape=(src, num_features))  # (batch, src, features)
    lstm_encoder = LSTM(64, return_sequences=True, dropout=0.2)(encoder_inputs)
    time_embedding_1 = Time2Vector(src)(lstm_encoder)
    x = Concatenate(axis=-1)([lstm_encoder, time_embedding_1])

    # Transformer Encoder Layer (1 layer)
    attn_output = MultiHeadAttention(num_heads=12, key_dim=64)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)
    x = TimeDistributed(Dense(ff_dim, activation='relu'))(x)
    x = TimeDistributed(Dense(512))(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    # Decoder
    decoder_inputs = Input(shape=(10, 2))  # (batch, 10, 2) - example 2D coords input
    lstm_decoder = LSTM(64, return_sequences=True, dropout=0.2)(decoder_inputs)
    time_embedding_2 = Time2Vector(10)(lstm_decoder)
    y = Concatenate(axis=-1)([lstm_decoder, time_embedding_2])

    # Transformer Decoder Layer (1 layer)
    attn1 = MultiHeadAttention(num_heads=12, key_dim=64)(y, y)
    attn2 = MultiHeadAttention(num_heads=12, key_dim=64)(attn1, x)
    y = LayerNormalization(epsilon=1e-6)(y + attn2)
    y = TimeDistributed(Dense(ff_dim, activation="relu"))(y)
    y = Dropout(0.1)(y)
    y = Add()([y, y])  # Residual connection (though this is doubling y; you might want y + previous input)
    y = LayerNormalization(epsilon=1e-6)(y)

    decoder_outputs = TimeDistributed(Dense(2, activation="linear"))(y)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mape'])
    return model

# usage:
if __name__ == "__main__":
    model = create_model(src=10, num_features=6)  # say 6 features: x, y, vx, vy, ax, ay
    model.summary()
