import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

# Use the GPU if you have one
if torch.cuda.is_available():
    # print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")
device = torch.device("cpu")
##################################################################################
######################## start of ptb-lm.py code #################################
##################################################################################

# we import some function given to us
def repackage_hidden(h):
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)
# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask

##################################################################################
######################## end of ptb-lm.py code ###################################
##################################################################################

##################################################################################
######################## start of code for prob 5.2 ##############################
##################################################################################

def depth_from_model(model, model_name):
    depth = 0
    if model_name == 'RNN' or model_name == 'GRU' :
        depth = model.num_layers
    elif model_name == 'TRANSFORMER':
        depth = len(model._modules['transformer_stack']._modules['layers'])
    else :
        pass
    return depth

def set_hooks(model, model_name):
    grads = {}
    def save_grad(name,signature):
        def hook(module, grad_input, grad_output):
            age = 1
            while (name,signature,age) in grads: # already there, add one more time step
                age += 1
            else :
                # old version : remember everything
                # grads[name,signature,age] = [grad_input, grad_output]
                grads[name,signature,age] = grad_output[0].detach().cpu()

        return hook

    list_grad_handle = []
    # for each model type in 'RNN', 'GRU', 'TRANSFORMER'
    # we place hooks at different place because their layers
    # are not named the same way and dont mean the same thing
    depth = depth_from_model(model, model_name)
    for i in range(depth) :
        if   model_name == 'RNN' :
            tmp = model.layers[i]._modules['0']
        elif model_name == 'GRU' :
            tmp = model.state_connections[i]._modules['0']
        elif model_name == 'TRANSFORMER':
            tmp = model._modules['transformer_stack']._modules['layers']._modules[str(i)]._modules['feed_forward']
        grad_handle  = tmp.register_backward_hook( save_grad('Loss grad', i) )
        list_grad_handle += [grad_handle]
    return grads, list_grad_handle

def run_batch_52(model, model_name, hidden, batch):
    """
    Takes a model and a batch
    returns a (model.num_layers x model.seq_len) tensor containing :
    the loss at time step -1 wrt the i hidden layer at time step j
    """
    # set hooks
    grads, list_grad_handle = set_hooks(model,model_name)
    # loss fct
    loss_fn = torch.nn.CrossEntropyLoss()
    model.zero_grad()
    # in training mode
    model.train()
    #
    # compute output and
    #   throw away hidden state
    #   only keep the last word prediction for each seq of the batch
    if model_name == 'TRANSFORMER':
        inputs      = Batch(torch.from_numpy(batch[0]).long().to(device))
        outputs     = model.forward(inputs.data, inputs.mask).transpose(1,0)
        lw_outputs  = outputs[-1,:,:].to(device)
    else :
        inputs      = torch.from_numpy(batch[0].astype(np.int64))
        inputs      = inputs.transpose(0, 1).contiguous().to(device)
        hidden      = repackage_hidden(hidden)
        outputs,_   = model.forward(inputs,hidden)
        lw_outputs  = outputs[-1,:,:].to(device)

    # for the targets, only keep the last word of each seq of a batch
    targets = torch.from_numpy(batch[1].astype(np.int64))
    targets = targets.transpose(0, 1).contiguous().to(device)[-1,:]
    tt      = torch.squeeze(targets.view(-1, model.batch_size ))

    # compute the loss but only for the last word of each seq of the batch
    loss    = loss_fn(lw_outputs.contiguous().view(-1, model.vocab_size ), tt)
    loss.backward()
    # now, because of the hooks, we can retrieve
    # the gradient of the loss wrt hidden layers for each time step

    depth  = depth_from_model(model, model_name)
    nablaL = torch.empty(model.batch_size, depth ,model.seq_len,model.hidden_size).cpu()

    # we retrieve gradients from hooks
    if model_name == 'TRANSFORMER':
        for i in range(depth):
            nablaL[:,i,:,:] = grads['Loss grad', i, 1]
    else:
        for i in range(depth):
            for j in range(model.seq_len):
                nablaL[:,i,j,:] = grads['Loss grad', i, model.seq_len-j]
    # now nablaL[k,i,j,l] contains :
    # the l component of the loss wrt the i hidden layer at time step j for sequence no k
    # we now compute the average norm across all sequence of the batch
    norm_nablaL     = torch.norm(nablaL,dim=-1)
    norm_avg_nablaL = torch.mean(norm_nablaL,dim=0).numpy()

    # now norm_avg_nablaL[i,j] contains :
    # the loss wrt the i hidden layer at time step j

    # remove hooks and empty dictionnary of gradients
    for grad_handle in list_grad_handle:
        grad_handle.remove()
    grads = {}

    # explicitely delete stuff that went to the gpu
    del inputs
    del outputs
    del lw_outputs
    del targets
    del tt

    return norm_avg_nablaL

def code_for_52(model, model_name, data):
    # if the model is recurent, we set its hidden state by
    # running the model on few batches
    iterator = ptb_iterator(data, model.batch_size, model.seq_len)
    if model_name == 'RNN' or model_name == 'GRU':
        hidden = model.init_hidden().to(device)
    else :
        hidden = None
    for i in range(5) :
        input, target = next(iterator,None)
        if model_name == 'RNN' or model_name == 'GRU':
            hidden = repackage_hidden(hidden)
            inputs = torch.from_numpy(input.astype(np.int64)).transpose(0, 1).contiguous().to(device)
            outputs, hidden = model(inputs, hidden)
    # define the batch we'll use
    input, target = next(iterator,None)
    batch = (input, target)
    # compute average norm of the loss at time step -1 across
    # hidden layers and time steps
    norm_avg_nablaL = run_batch_52(model, model_name, hidden, batch)

    # do other thing if you want e.g. plot
    return norm_avg_nablaL

##################################################################################
########################## end of code for prob 5.2 ##############################
##################################################################################

##################################################################################
######################## start of code for prob 5.1 ##############################
##################################################################################

def code_for_51(model, model_name, force_init, data):
    """
    Input  : a model, its name, and the data
             force_init is either true or false and
             is set to true to re-initialize hidden state
             at the start of each new batch
    output : a numpy array containing the loss
             averaged across sequence of the same minibatch
             for each time step and for each minibatch
    """
    # run in eval mode
    model.eval()
    # loss fct
    loss_fn = torch.nn.CrossEntropyLoss()
    #
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_len
    avg_error  = np.empty([epoch_size,model.seq_len])
    if model_name != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)
    # no gradient here
    with torch.no_grad():
        for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
            if model_name == 'TRANSFORMER':
                batch = Batch(torch.from_numpy(x).long().to(device))
                outputs = model.forward(batch.data, batch.mask).transpose(1,0)
            else:
                if force_init is True:
                    hidden = model.init_hidden()
                    hidden = hidden.to(device)
                inputs = torch.from_numpy(x.astype(np.int64))
                inputs = inputs.transpose(0, 1).contiguous().to(device)
                model.zero_grad()
                hidden = repackage_hidden(hidden)
                outputs, hidden = model(inputs, hidden)

            targets = torch.from_numpy(y.astype(np.int64))
            targets = targets.transpose(0, 1).contiguous().to(device)

            # LOSS COMPUTATION FOR PROB 5.1
            # We compute the average loss at each time-step separately.
            # for the tensor targets and outputs we have that
            #   dim 0 iterates across time steps (i.e. words) in a sequence
            #   dim 1 iterates across sequences in a batch
            # this means that we have to compute the loss across dim 0
            # and store it
            outputs = outputs.contiguous().to(device)
            # print( outputs.shape )
            # print( targets.shape )
            # print( avg_error.shape )
            for t in range(model.seq_len):
                # the default behavior of CrossEntropyLoss is to use
                # reduction='mean' i.e. average over loss elements
                # and that is what we want
                loss = loss_fn(outputs[t,:], targets[t,:])
                avg_error[step,t] = loss

    # free memory
    del x
    del y
    # do other thing if you want e.g. plot
    return avg_error

##################################################################################
########################## end of code for prob 5.1 ##############################
##################################################################################
