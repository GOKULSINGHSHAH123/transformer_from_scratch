import torch
import torch.nn as nn
import math


# When the tokens are passed through this function they are converted to vectors
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

# Now we will do positional encodings. 
# This means now we add a vector that will represent their position.

# pe(pos,2i) = sin(pos/10000^2i/d_model)
# pe(pos,2i+1) = sin(pos/10000^2i/d_model)

# pos: the position of the vector we want to find 
# i: ranges from 0 to (d_model/2) - 1
# d_model: vector size, taking 512
# This creates a positional encoding of a position of a vector of size 512 that we will 
# add in input embeddings.

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].unsqueeze(0)
        return self.dropout(x)
    
    
# Layer Normalisation is used after output of each layer.
# It also helps to prevent vanishing gradient problem.
# It helps to more stable and efficient training. 
# After applying means and variance,
# there are alpha and gamma trainable parameters also.

class LayerNormalization(nn.Module):

    def __init__(self, d_model: int, eps: float= 10**-6) ->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # multiplied
        self.bias = nn.Parameter(torch.zeros(d_model)) # added

    def forward(self, x):
        mean = x.mean(dim =-1,keepdim= True)
        std = x.std(dim = -1,keepdim =True)
        return self.alpha*(x-mean)/(std +self.eps) +self.bias
        

# Creating feed forward neural network  
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
         # Here we are defining the neural network layers
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        # (Batch, Seq_length, d_model) --> (Batch, Seq_lenth, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# Creating multi-head self-attention
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int ,h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 , "d_model is not divisible by h"

        # Creating neural network layers to train and get q k v 
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv
        
        # This is also neural network applied to end of the output
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores =  (query @ key.transpose(-2,-1))/ math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0 ,-1e9) 

        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_lenth, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value) , attention_scores

    def forward(self, q, k, v, mask):
        # Creating trainables q, k, v by neural networks
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Shaping them for future mathematics to achieve self-attention
        query = query.view(query.shape[0] ,query.shape[1], self.h , self.d_k).transpose(1,2)
        key = key.view(query.shape[0] ,query.shape[1], self.h , self.d_k).transpose(1,2)
        value = value.view(query.shape[0] ,query.shape[1], self.h , self.d_k).transpose(1,2)

        # Now passing two attention function to get attention scores
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Now reshaping again to original form
        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        return self.w_o(x) 
    
class ResidualConnection(nn.Module):

    def __init__ (self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Combining all of the code to encoder block    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])


    # these code are like the procedure of encoder block
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)

        return x
    
 # This is the code for n number of encoder layers   
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(layers[0].self_attention_block.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

# Now making a decoder block

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(layers[0].self_attention_block.d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# Now making the vectors linear and applying softmax for prediction
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embedding
        self.target_embeddings = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.target_embeddings(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    src_embedded = InputEmbeddings(d_model, src_vocab_size)
    tgt_embedded = InputEmbeddings(d_model, tgt_vocab_size)

    # Creating the positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = nn.ModuleList([
        EncoderBlock(
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout
        ) for _ in range(N)
    ])

    # Create the decoder blocks
    decoder_blocks = nn.ModuleList([
        DecoderBlock(
            MultiHeadAttentionBlock(d_model, h, dropout),
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout
        ) for _ in range(N)
    ])
    
    # Create the encoder and decoder
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embedded, tgt_embedded, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
