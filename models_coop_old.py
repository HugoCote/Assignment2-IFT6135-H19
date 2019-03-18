import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt


# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is
# what the main script expects. If you modify the contract,
# you must justify that choice, note it in your report, and notify the TAs
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention.


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.

    def __init__(self, emb_size, hidden_size, seq_len, batch_size,
               vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The numvwe of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """

        super(RNN, self).__init__()

        # Attributes
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.emb_size = emb_size
        self.num_layers = num_layers

        # Embedding layer
        self.embeddings = WordEmbedding(emb_size, vocab_size)

        # Dropout after embedding
        self.dropout_layer = nn.Dropout(1-dp_keep_prob)

        # First layer: receives emb_size and hidden size. Output hidden_size
        self.first_hidden_layer = nn.Linear(emb_size + hidden_size, hidden_size, bias=True)

        # Hidden layer: receives hidden and lag hidden Output hidden_size
        self.hidden_layer = nn.Linear(hidden_size + hidden_size, hidden_size, bias=True)

        # Output layer: receives hidden size
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=True)

        # Input to hidden layer
        first_seq = nn.Sequential(
            self.first_hidden_layer,
            nn.Tanh(),
            nn.Dropout(1-dp_keep_prob),
        )

        # hidden to hidden layer
        hidden_seq = nn.Sequential(
            self.hidden_layer,
            nn.Tanh(),
            nn.Dropout(1-dp_keep_prob),
        )

        # Stack the hidden layers with same shape (n_layers minus 1 because
        # the first layer has different input size)
        self.layers = clones(hidden_seq, self.num_layers-1)

        # Adjust for first layer.
        self.layers = nn.ModuleList([first_seq, *self.layers])

        # Initialize weights
        self.init_weights_uniform()

        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers.
        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the
        # provided clones function (as opposed to a regular python list), in order
        # for Pytorch to recognize these parameters as belonging to this nn.Module
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.

    def init_weights_uniform(self):
        # TODO ========================
        # Initialize all the weights uniformly in the range [-0.1, 0.1]
        # and all the biases to 0 (in place)

        embeddings = self.embeddings.lut
        embeddings.weight.data = torch.Tensor(embeddings.weight.shape).uniform_(-0.1, 0.1)
        # embedding.bias.data = torch.zeros(embedding.bias.shape)
        self.embeddings = embeddings

        # # For the current time step, go through the all the hidden layers
        # for layer in range(0, self.num_layers):
        #     print("LAYER: {}".format(layer))
        #     tmp_weight_shape = self.layers[layer][0].weight.shape
        #     self.layers[layer][0].weight.data = torch.Tensor(tmp_weight_shape).uniform_(-0.1, 0.1)
        #
        #     tmp_bias_shape = self.layers[layer][0].bias.shape
        #     self.layers[layer][0].bias.data = torch.zeros(tmp_bias_shape)

        # Output
        self.output_layer.weight.data = torch.Tensor(self.output_layer.weight.shape).uniform_(-0.1, 0.1)
        self.output_layer.bias.data = torch.zeros(self.output_layer.bias.shape)

    def init_hidden(self):
        # TODO ========================
        """
        initialize the hidden states to zero
        This is used for the first mini-batch in an epoch, only.
        """
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        return h0  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)


    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using a nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """

        # TODO Initialize layers
        # HERE

        # Set logits to zero (initialization)
        logits = torch.zeros(self.seq_len, self.batch_size, self.vocab_size)

        # TODO I am not taking into account batch size here. Not sure how to do it.

        # Loop through time.
        for step in range(0, self.seq_len):

            # Input is (seq_len, batch_size)
            inp = inputs[step]
            # Embedding output the for word at time t.
            embed = self.embeddings(inp)

            # Dropout after embedding
            self.dropout_layer(embed)

            # First hidden layer output
            h = hidden[0]  # => shape: (1, batch_size, hidden_size)

            # embed + h as input for the next time step
            combined = torch.cat((h, embed), 1)

            # Forward through first layer
            out = self.layers[0](combined)

            # This will be used in the next timestep as the lag value.
            hidden[0] = out

            # For the current time step, go through the all the hidden layers
            for layer in range(1, self.num_layers):

                # Take the hidden state of the current layer
                h = hidden[layer]

                # Combine hidden state of l-th layer and current hidden state
                combined = torch.cat((h, out), 1)
                out = self.layers[layer](combined)

                # This will be used in the next timestep as the lag value.
                hidden[layer] = out

            # last layer to calculate the logits
            # not sure here when they say not to do softmax.... ???
            logits[step] = self.output_layer(out)

        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def sample_from_density(self, density_matrix):
        # input : density_tensor :
        # dim -1 of this tensor represent a valid probability mass function
        # last dim. of the input tensor will be shrinked and squeezed to produce the output
        # i.e. density_matrix has shape n_1 x n_2 and output had shape n_1
        # output contains sampled indices
        rand_ind = torch.multinomial(density_matrix, 1, replacement=True)
        return rand_ind

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        # TODO Initialize layers
        # HERE

        # Initilize samples
        # the entries of this matrix will be computed sequentialy row by row
        samples = torch.empty( generated_seq_len , self.batch_size, dtype=torch.long)
        samples[0,:] = input

        # Loop through time.
        for step in range(1, generated_seq_len + 1):

            inp = samples[step-1,:]
            # Embedding output the for word at time t.
            embed = self.embeddings(inp)

            # Dropout after embedding
            self.dropout_layer(embed)

            # First hidden layer output
            h = hidden[0]  # => shape: (1, batch_size, hidden_size)

            # embed + h as input for the next time step
            combined = torch.cat((h, embed), 1)

            # Forward through first layer
            out = self.layers[0](combined)

            # This will be used in the next timestep as the lag value.
            hidden[0] = out

            # For the current time step, go through the all the hidden layers
            for layer in range(1, self.num_layers):

                # Take the hidden state of the current layer
                h = hidden[layer]

                # Combine hidden state of l-th layer and current hidden state
                combined = torch.cat((h, out), 1)
                out = self.layers[layer](combined)

                # This will be used in the next timestep as the lag value.
                hidden[layer] = out

            # last layer to calculate the logits
            # not sure here when they say not to do softmax.... ???
            logits[step] = self.output_layer(out)

        # return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

        return samples


class GRU(nn.Module):  # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for
    GRU, not Vanilla RNN.
    """

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        # word embedding
        self.embedding = WordEmbedding(emb_size, vocab_size)

        # hidden layers:
        self.hidden_gates = nn.ModuleList()  # compute the update and reset gates
        self.hidden_states = nn.ModuleList()  # compute the cell state using the reset gate
        for i in range(num_layers):
            if i == 0:
                input_size = emb_size + hidden_size
            else:
                input_size = 2 * hidden_size
            self.hidden_gates.append(nn.Sequential(
                nn.Linear(input_size, 2 * hidden_size),
                nn.Sigmoid()
            ))
            self.hidden_states.append(nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh()
            ))

        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def init_weights_uniform(self):
        pass

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        logits = []
        embeddings = self.embedding(inputs)
        for i in range(self.seq_len):
            hiddens = []
            layer_inpt = embeddings[i]
            for n in range(self.num_layers):
                hidden_gates = self.hidden_gates[n](torch.cat([layer_inpt, hidden[n]], dim=1))
                update_gate = hidden_gates[:, :self.hidden_size]
                reset_gate = hidden_gates[:, self.hidden_size:]

                cell_state = self.hidden_states[n](torch.cat([layer_inpt, reset_gate * hidden[n]], dim=1))

                hidden_out = (1. - update_gate) * hidden[n] + update_gate * cell_state

                hiddens.append(hidden_out.unsqueeze(0))
                layer_inpt = hidden_out  # input for the next layer

            output = self.output_layer(hidden_out)
            hidden = torch.cat(hiddens, dim=0)
            logits.append(output)

        logits = torch.cat(logits, dim=0)
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        samples = None
        return samples


# class LSTM(nn.Module):
#     def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
#         super(GRU, self).__init__()
#         self.emb_size = emb_size
#         self.hidden_size = hidden_size
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.vocab_size = vocab_size
#         self.num_layers = num_layers
#         self.dp_keep_prob = dp_keep_prob
#
#         # word embedding
#         self.embedding = WordEmbedding(emb_size, vocab_size)
#
#         # hidden layers:
#         self.hidden_layers = nn.ModuleList()
#         for i in range(num_layers):
#             if i == 0:
#                 self.hidden_layers.append(nn.Linear(emb_size + hidden_size, 4 * hidden_size))
#             else:
#                 self.hidden_layers.append(nn.Linear(2 * hidden_size, 4 * hidden_size))
#
#     def init_hidden(self):
#         return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
#
#     def forward(self, inputs, hidden):
#         logits = []
#         embeddings = self.embedding(inputs)
#         for i in range(self.seq_len):
#             for n in range(self.num_layers):
#                 if n == 0:
#                     layer_inpt = embeddings[i]
#                 else:
#                     layer_inpt = hidden_gate_out
#                 hidden_gate_out = self.hidden_layers[n](torch.cat([layer_inpt, hidden[n]], dim=1))  # input, output, forget gates
#                 input_gate = F.sigmoid(hidden_gate_out[:self.hidden_size])
#                 forget_gate = F.sigmoid(hidden_gate_out[self.hidden_size: 2 * self.hidden_size])
#                 output_gate = F.sigmoid(hidden_gate_out[2 * self.hidden_size: 3 * self.hidden_size])
#                 cell_context = F.tanh(hidden_gate_out[3 * self.hidden_size:])
#                 cell_state =
#                 hidden[n] = hidden_gate_out
#             output = self.output_layer(hidden_gate_out)
#             logits.append(output)
#
#         logits = torch.cat(logits, dim=0)
#         return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

# ----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        # print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# ----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # apply the self-attention
        return self.sublayer[1](x, self.feed_forward)  # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """

    def __init__(self, layer, n_blocks):  # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


# def make_model(vocab_size, n_blocks=6, n_units=512, n_heads=16, dropout=0.1):
#     "Helper: Construct a model from hyperparameters."
#     c = copy.deepcopy
#     attn = MultiHeadedAttention(n_heads, n_units)
#     ff = MLP(n_units, dropout)
#     position = PositionalEncoding(n_units, dropout)
#     model = FullTransformer(
#         transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
#         embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
#         n_units=n_units,
#         vocab_size=vocab_size
#         )
#
#     # Initialize parameters with Glorot / fan_avg.
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model


# ----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# ----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """

    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
