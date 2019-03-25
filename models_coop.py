import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import torch.nn.init as init

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
class RNN(nn.Module):  # Implement a stacked vanilla RNN with Tanh nonlinearities.

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

        # Dropout
        self.dp = nn.Dropout(1. - dp_keep_prob)

        # Output layer: receives hidden size
        self._output_layer = nn.Linear(hidden_size, vocab_size, bias=True)

        # Init embedding
        self.init_weights_uniform()

        # Build the stack
        self.layers = nn.ModuleList()
        input_size = emb_size
        for _ in range(num_layers):
            mod_list = nn.ModuleList()
            mod_list.append(nn.Linear(hidden_size, hidden_size, bias=True))
            mod_list.append(nn.Linear(input_size, hidden_size, bias=True))
            mod_list.append(nn.Tanh())
            input_size = hidden_size
            self.layers.append(mod_list)

    def init_weights_uniform(self):
        # Initialize all the weights uniformly in the range [-0.1, 0.1]
        # and all the biases to 0 (in place)
        init.uniform_(self.embeddings.lut.weight, -0.1, 0.1)
        init.uniform_(self._output_layer.weight, -0.1, 0.1)
        with torch.no_grad():
            self._output_layer.bias.zero_()

    def init_hidden(self):
        """
        initialize the hidden states to zero
        This is used for the first mini-batch in an epoch, only.
        """
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        return h0  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
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
        # Loop through time.
        logits = []
        for step in range(0, self.seq_len):

            # Input is (seq_len, batch_size)
            inp = inputs[step]

            # Embedding output the for word at time t.
            out = self.embeddings(inp)

            # step hidden to avoid issue
            step_hidden = []
            # For the current time step, go through the all the hidden layers
            for layer in range(0, self.num_layers):
                # Take the hidden state of the current layer

                hidden2hidden, layer2layer, tanh = self.layers[layer]

                # Get hidden t-1
                h = hidden[layer]

                # Previous hidden
                h = hidden2hidden(h)

                # Dropout before weights
                out = self.dp(out)

                # Previous layer to next layer
                out = layer2layer(out)

                # h + out before tanh
                out += h

                # tanh
                out = tanh(out)

                step_hidden.append(out)

            hidden = step_hidden

            # last layer to calculate the logits
            # Is it fine to do dropout here as well ?
            out = self.dp(out)
            out = self._output_layer(out)
            logits.append(out)

        logits = torch.cat(logits).view(self.seq_len, self.batch_size, self.vocab_size)

        return logits, hidden

    def sample_from_logits(self, logits_matrix):
        # input : logits_tensor :
        # dim -1 of this tensor represent a valid temperature function
        # last dim. of the input tensor will be shrinked and squeezed to produce the output
        # i.e. logits_matrix has shape n_1 x n_2 and output had shape n_1
        # output contains sampled indices and their (unaltered) log-probabilities

        # step 1 : increase the temperature if necesary to over-sample
        # from the most probable token or do the reverse
        temp = 1.5
        density_matrix     = F.softmax( temp * logits_matrix , dim=-1)
        log_density_matrix = F.log_softmax( logits_matrix , dim=-1)

        # hardcoded : prevend choosing '<unk>' label by setting prob to zero
        bs = density_matrix.shape[0]
        density_matrix[:,1] = torch.zeros(bs)

        rand_ind = torch.multinomial(density_matrix, 1, replacement=True)
        # gather true prob of sampled token
        probs    = torch.gather(log_density_matrix , 1, rand_ind)
        # return sample and their prob
        return rand_ind.squeeze(-1), probs.squeeze(-1)

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
            - the log-probability of sampled sequences
        """
        in_batch_size = input.shape[0]
        samples = torch.empty( generated_seq_len , in_batch_size, dtype=torch.long)
        logprob = torch.zeros( in_batch_size, dtype=torch.float)
        samples[0,:] = input
        for step in range(1, generated_seq_len):

            # Input is (seq_len, batch_size)
            inp = samples[step-1,:]

            # Embedding output the for word at time t.
            out = self.embeddings(inp)

            # step hidden to avoid issue
            step_hidden = []
            # For the current time step, go through the all the hidden layers
            for layer in range(0, self.num_layers):
                # Take the hidden state of the current layer

                hidden2hidden, layer2layer, tanh = self.layers[layer]

                # Get hidden t-1
                h = hidden[layer]

                # Previous hidden
                h = hidden2hidden(h)

                # Dropout before weights
                out = self.dp(out)

                # Previous layer to next layer
                out = layer2layer(out)

                # h + out before tanh
                out += h

                # tanh
                out = tanh(out)

                step_hidden.append(out)

            hidden = step_hidden

            # last layer to calculate the logits
            out = self.dp(out)
            out = self._output_layer(out)
            # convert logits to density and sample from it
            # output_layer returns (in_batch_size, self.vocab_size)
            # sampled_indices is (in_batch_size) and serves has input to next step
            sampled_indices, probs    = self.sample_from_logits( out )
            samples[ step , :] = sampled_indices
            logprob            = logprob + probs

        return samples, logprob, hidden

# Problem 2
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
        self.embeddings = WordEmbedding(emb_size, vocab_size)
        self.dropout = nn.Dropout(1. - dp_keep_prob)
        self.tanh = nn.Tanh()


        self.reset_gates = nn.ModuleList()
        self.update_gates = nn.ModuleList()
        self.state_connections = nn.ModuleList()
        input_size = emb_size
        for i in range(num_layers):
            # reset gate
            connections = nn.ModuleList()
            connections.append(nn.Linear(input_size, hidden_size))
            connections.append(nn.Linear(hidden_size, hidden_size))
            connections.append(nn.Sigmoid())
            self.reset_gates.append(connections)

            # update gate
            connections = nn.ModuleList()
            connections.append(nn.Linear(input_size, hidden_size))
            connections.append(nn.Linear(hidden_size, hidden_size))
            connections.append(nn.Sigmoid())
            self.update_gates.append(connections)

            # hidden, input to hidden connection
            connections = nn.ModuleList()
            connections.append(nn.Linear(input_size, hidden_size))
            connections.append(nn.Linear(hidden_size, hidden_size))
            connections.append(nn.Tanh())
            self.state_connections.append(connections)

            input_size = hidden_size

        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.init_weights_uniform()


    def init_weights_uniform(self):
        # std = 1./(self.hidden_size ** 0.5)
        # for w in self.parameters():
        #     init.uniform_(w, -std, std)
        init.uniform_(self.embeddings.lut.weight, -0.1, 0.1)
        init.uniform_(self.output_layer.weight, -0.1, 0.1)
        with torch.no_grad():
            self.output_layer.bias.zero_()


    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        outputs = []
        for n in range(self.seq_len):
            input = self.embeddings(inputs[n])
            new_hiddens = []
            for l in range(self.num_layers):
                i_r, h_r, activation = self.reset_gates[l]
                reset_gate = activation(i_r(input) + h_r(hidden[l]))
                i_u, h_u, activation = self.update_gates[l]
                update_gate = activation(i_u(input) + h_u(hidden[l]))
                i_c, h_c, activation = self.state_connections[l]
                cell_intermediate_state = activation(i_c(self.dropout(input)) + reset_gate * h_c(hidden[l]))
                cell_state = (1. - update_gate) * cell_intermediate_state + update_gate * hidden[l]
                new_hiddens.append(cell_state)
                input = cell_state

            output = self.output_layer(self.dropout(cell_state))
            # output = self.output_layer(cell_state)
            outputs.append(output)
            hidden = new_hiddens

        outputs = [o.unsqueeze(0) for o in outputs]
        hidden = [h.unsqueeze(0) for h in hidden]
        return torch.cat(outputs, dim=0), torch.cat(hidden, dim=0)


    def sample_from_logits(self, logits_matrix):
        # input : logits_tensor :
        # dim -1 of this tensor represent a valid temperature function
        # last dim. of the input tensor will be shrinked and squeezed to produce the output
        # i.e. logits_matrix has shape n_1 x n_2 and output had shape n_1
        # output contains sampled indices and their (unaltered) log-probabilities

        # step 1 : increase the temperature if necesary to over-sample
        # from the most probable token
        temp = 1.5
        density_matrix     = F.softmax( temp * logits_matrix , dim=-1)
        log_density_matrix = F.log_softmax( logits_matrix , dim=-1)

        # hardcoded : prevend choosing '<unk>' label by setting prob to zero
        bs = density_matrix.shape[0]
        density_matrix[:,1] = torch.zeros(bs)

        rand_ind = torch.multinomial(density_matrix, 1, replacement=True)
        # gather true prob of sampled token
        probs    = torch.gather(log_density_matrix , 1, rand_ind)
        # return sample and their prob
        return rand_ind.squeeze(-1), probs.squeeze(-1)

    def generate(self, inputs, hidden, generated_seq_len):
        """
        Arguments:
            - inputs: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
            - the log-probability of sampled sequences
        """
        samples = torch.empty( generated_seq_len , self.batch_size, dtype=torch.long)
        logprob = torch.zeros( self.batch_size, dtype=torch.float)
        samples[0,:] = inputs

        for n in range( 1 , generated_seq_len ):
            input = self.embeddings( samples[n-1,:] )
            new_hiddens = []
            for l in range(self.num_layers):
                i_r, h_r, activation = self.reset_gates[l]
                reset_gate = activation(i_r(input) + h_r(hidden[l]))
                i_u, h_u, activation = self.update_gates[l]
                update_gate = activation(i_u(input) + h_u(hidden[l]))
                i_c, h_c, activation = self.state_connections[l]
                cell_intermediate_state = activation(i_c(self.dropout(input)) + reset_gate * h_c(hidden[l]))
                cell_state = (1. - update_gate) * cell_intermediate_state + update_gate * hidden[l]
                new_hiddens.append(cell_state)
                input = cell_state

            output = self.output_layer(self.dropout(cell_state))
            # output = self.output_layer(cell_state)
            sampled_indices, probs = self.sample_from_logits( output )
            samples[ n , :] = sampled_indices
            logprob            = logprob + probs
            hidden = new_hiddens

        hidden = [h.unsqueeze(0) for h in hidden]

        return samples, logprob, torch.cat(hidden, dim=0)


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
