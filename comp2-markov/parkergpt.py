import tensorflow as tf
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Layer, LayerNormalization, Dropout
from tensorflow.keras.models import Model

class JazzTransformer(Model):
    def __init__(self, num_layers=6, d_model=512, num_heads=8, dff=2048):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff)
        self.forward_head = Dense(token_vocab_size)
        self.backward_head = Dense(token_vocab_size)
        
    def call(self, inputs):
        # Input shape: (batch_size, seq_len, d_model)
        enc_output = self.encoder(inputs)
        dec_output = self.decoder(enc_output)
        
        return {
            "forward": self.forward_head(dec_output),
            "backward": self.backward_head(dec_output)
        }

class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff):
        super().__init__()
        self.layers = [
            TransformerBlock(  # Custom layer
                d_model, num_heads, dff,
                bidirectional_attention=True  # ← Key modification
            ) for _ in range(num_layers)
        ]
    
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff):
        super().__init__()
        self.layers = [
            TransformerBlock(  # Custom layer
                d_model, num_heads, dff,
                bidirectional_attention=True
            ) for _ in range(num_layers)
        ]

        self.forward_mha = MultiHeadAttention(num_heads, d_model)
        self.backward_mha = MultiHeadAttention(num_heads, d_model)
        self.switch = Dense(2)  # Learned blending ratio
    
    def call(self, enc_output):
        # Parallel forward/backward attention
        fwd = self.forward_mha(enc_output)
        bwd = self.backward_mha(enc_output)
        
        # Dynamic blending
        blend_weights = tf.nn.softmax(self.switch(enc_output))
        return blend_weights[..., 0:1] * fwd + blend_weights[..., 1:2] * bwd

class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, dff, bidirectional_attention=False, dropout_rate=0.1):
        super().__init__()
        
        self.bidirectional = bidirectional_attention
        self.mha = MultiHeadAttention(num_heads, d_model)
        if self.bidirectional:
            self.backward_mha = MultiHeadAttention(num_heads, d_model)
#TODO: define the rest of the TransformerBlock, augmented with bidirectional attention

def jazz_loss(y_true, y_pred):
    # TODO: Forward prediction loss
    
    # TODO: Backward prediction loss

# Modified sinusoidal encoding
class PhraseAwarePE(tf.keras.layers.Layer):
    def __init__(self, max_phrase_length=8):
        super().__init__()
#TODO: define phrase aware positional encoding