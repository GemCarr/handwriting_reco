
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn
from datetime import datetime
from sklearn.utils import shuffle
import random
import torch.nn.functional as F
import editdistance
import math

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomLSTM(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, n_layers: int, batch_first = True, bidirectional=False) -> None:
        """
        Args:
            embed_size: embedding dimensions.
            hidden_size: hidden layer size.
            n_layers: the number of layers.
            n_outputs: the number of output classes.
        """
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        # The LSTM
        self.lstm = nn.LSTM(
            input_size = self.embed_size,
            hidden_size = self.hidden_size,
            num_layers = self.n_layers,
            batch_first = self.batch_first, # Changes the order of dimension to put the batches first.
            bidirectional = self.bidirectional,
        )

    def compute_pca_matrix(self, X, pca_sample_size, regularization = 10**-5, normalize=False):
        pca_weight_list = []
        pca_bias_list = []

        remaining_dim = self.hidden_size * 4
       # random_dim = self.hidden_size - remaining_dim
        random_dim = 0

        while(remaining_dim > 0):

            pca_vectors = X[torch.randperm(X.shape[0])[:pca_sample_size]]
            
            _, s, v = torch.pca_lowrank(pca_vectors, q=X.shape[1]//8)


            mean = torch.mean(X, dim=0)

            if normalize:
            
                explained_variance = (s ** 2) / pca_sample_size
                v = v / torch.sqrt(explained_variance + regularization)
           
            pca_weight = v.transpose(0,1)
            pca_bias = torch.matmul(mean, -v)

            if remaining_dim < X.shape[1]//8:
                pca_weight = pca_weight[:remaining_dim, :]
                pca_bias = pca_bias[:remaining_dim]


            pca_weight_list.append(pca_weight)
            pca_bias_list.append(pca_bias)

            remaining_dim -= X.shape[1]//8
        
        if (random_dim > 0):
            random_weights = torch.empty(random_dim, X.shape[1])
            random_biases = torch.empty(random_dim)
            k = 1 / self.hidden_size
            torch.nn.init.uniform_(random_weights, a=-math.sqrt(k), b=math.sqrt(k))
            torch.nn.init.uniform_(random_biases, a=-math.sqrt(k), b=math.sqrt(k))

            pca_weight_list.append(random_weights)
            pca_bias_list.append(random_biases)

        return torch.cat(pca_weight_list), torch.cat(pca_bias_list)


    def pca_weight_init(self, X, pca_sample_size) :
        with torch.no_grad():
            pca_weight, pca_bias = self.compute_pca_matrix(X, pca_sample_size, normalize=True)

            h = torch.zeros(X.size(0), self.hidden_size).to(device)
            c = torch.zeros(X.size(0), self.hidden_size).to(device)
            hs = self.hidden_size
            ins = self.embed_size
            hn = X
            
            parameters = self.lstm.named_parameters()

            for _ in range(self.n_layers):

                weight_ih = next(parameters)[1].data
                weight_hh = next(parameters)[1].data 
                bias_ih = next(parameters)[1].data
                bias_hh = next(parameters)[1].data


                weight_ih[:,:] = pca_weight
                bias_ih[:] = pca_bias
                

                cell = nn.LSTMCell(input_size=ins, hidden_size=hs).to(device)
                ins = hs * 2 if self.bidirectional else hs


                cell.weight_ih[:,:] = pca_weight
                cell.bias_ih[:] = pca_bias
                

                cell.weight_hh[:,:] = weight_hh[:,:]
                cell.bias_hh[:] = bias_hh[:]

                hn, _ = cell(hn, (h, c))
                pca_weight2, pca_bias2 = self.compute_pca_matrix(hn, pca_sample_size)

                weight_hh[:,:] = pca_weight2
                bias_hh[:] = pca_bias2
                
                if self.bidirectional :
                    weight_ih_reverse = next(parameters)[1].data
                    weight_hh_reverse = next(parameters)[1].data 
                    bias_ih_reverse = next(parameters)[1].data
                    bias_hh_reverse = next(parameters)[1].data

                    weight_ih_reverse[:,:] = pca_weight
                    bias_ih_reverse[:] = pca_bias

                    weight_hh_reverse[:,:] = pca_weight2
                    bias_hh_reverse[:] = pca_bias2
                    


                    hn = torch.cat([hn]*2, dim=1).to(device)
                    pca_weight2, pca_bias2 = self.compute_pca_matrix(hn, pca_sample_size)


                pca_weight = pca_weight2
                pca_bias = pca_bias2


    def forward(self, X: torch.Tensor, init_state) -> torch.Tensor:
        return self.lstm.forward(X, init_state)


class CustomLSTMClassifier(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, n_layers: int, n_outputs: int, pca_data=None, pca_sample_size=1000) -> None:
        """
        Args:
            embed_size: embedding dimensions.
            hidden_size: hidden layer size.
            n_layers: the number of layers.
            n_outputs: the number of output classes.
        """
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_outputs = n_outputs

        # The LSTM
        self.lstm = CustomLSTM(
            embed_size = self.embed_size,
            hidden_size = self.hidden_size,
            n_layers = self.n_layers,
            bidirectional=True
        )

        if pca_data is not None and pca_sample_size > 0:
            self.lstm.pca_weight_init(pca_data, pca_sample_size)
        

        # A fully connected layer to project the LSTM's output to only one output used for classification.
        self.fc = nn.Linear(2 * self.hidden_size, self.n_outputs)
  
    def pca_weight_init(self,pca_data, pca_sample_size):
        self.lstm.pca_weight_init(pca_data, pca_sample_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Function called when the model is called with data as input.
        Args:
            X: the input tensor of dimensions batch_size, sequence length, word embedding size
        """
        h0 = torch.zeros(self.n_layers*2, X.size(0), self.hidden_size).to(device)
        
        # we create tensors for storing the initial LSTM cell states
        c0 = torch.zeros(self.n_layers*2, X.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(X, (h0, c0))

        # Getting the last value only.
        out = out[:, -1, :]
    
        # Linear projection.
        out = self.fc(out)

        return out

class CustomLSTMCTC(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, n_layers: int, n_outputs: int, use_embed=False, pca_data=None, pca_sample_size = 0) -> None:
        """
        Args:
            embed_size: embedding dimensions.
            hidden_size: hidden layer size.
            n_layers: the number of layers.
            n_outputs: the number of output classes.
        """
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.use_embed = use_embed

        # The LSTM
        self.lstm = CustomLSTM(
            embed_size = self.hidden_size if self.use_embed else self.embed_size,
            hidden_size = self.hidden_size,
            n_layers = self.n_layers,
            batch_first=False,
            bidirectional=True
        )

        if pca_data is not None and pca_sample_size > 0:
            self.lstm.pca_weight_init(pca_data, pca_sample_size)
        
        self.embed = torch.nn.Linear(self.embed_size, self.hidden_size)

        # A fully connected layer to project the LSTM's output to only one output used for classification.
        self.fc = nn.Linear(2 * self.hidden_size, self.n_outputs)
 
    def pca_weight_init(self,pca_data, pca_sample_size):
        self.lstm.pca_weight_init(pca_data, pca_sample_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Function called when the model is called with data as input.
        Args:
            X: the input tensor of dimensions batch_size, sequence length, word embedding size
        """
        h0 = torch.zeros(self.n_layers*2, X.size(1), self.hidden_size).to(device)
        
        # we create tensors for storing the initial LSTM cell states
        c0 = torch.zeros(self.n_layers*2, X.size(1), self.hidden_size).to(device)


        if self.use_embed:
            X = self.embed(X)
        out, _ = self.lstm(X, (h0, c0))
        

        # Linear projection.
        out = self.fc(out)

        return out





def data_generator(X: np.ndarray, y: np.ndarray, batch_size: int = 32, embedding_size=100, pad_right: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Generate randomly ordered batches of data+labels.
    Args:
        X: the input data.
        y: the corresponding labels.
        batch_size: the size of each batch [32].
        pad_right: if true, the padding is done on the right [False].
    """
    
    X, y = shuffle(X, y)
    n_batches = int(np.ceil(len(y) / batch_size))
    
    for i in range(n_batches):
        
        end = min((i+1)*batch_size, len(y))
        
        X_batch = X[i*batch_size:end]
        y_batch = y[i*batch_size:end]

        # Padding to max length size within the batch
        max_len = np.max([len(x) for x in X_batch])

        X_batch_res = np.zeros((len(y_batch), max_len, embedding_size))


        for j in range(len(X_batch)):
            x = X_batch[j]

            if max_len - len(x) > 0:
                pad = np.array([([0] * embedding_size)] * (max_len - len(x)))
                X_batch_res[j] = np.concatenate((x, pad)) if pad_right else np.concatenate((pad, x))
            else:
                X_batch_res[j] = x

        X_batch = torch.from_numpy(np.array(X_batch_res)).float()
        y_batch = torch.from_numpy(np.array(y_batch)).float()
        yield X_batch, y_batch

def data_generator_ctc(X: np.ndarray, y: np.ndarray, converter, batch_size: int = 32, embedding_size=64) -> Tuple[np.ndarray, np.ndarray]:
    """Generate randomly ordered batches of data+labels.
    Args:
        X: the input data.
        y: the corresponding labels.
        batch_size: the size of each batch [32].
        pad_right: if true, the padding is done on the right [False].
    """
    
    X, y = shuffle(X, y)
    n_batches = int(np.ceil(len(y) / batch_size))
    
    for i in range(n_batches):
        
        end = min((i+1)*batch_size, len(y))
        
        X_batch = X[i*batch_size:end]
        y_batch = y[i*batch_size:end]

        # Padding to max length size within the batch
        max_len = np.max([len(x) for x in X_batch])

        X_batch_res = np.zeros((len(y_batch), max_len, embedding_size))

        X_batch_len = []
        y_batch_len = []


        for j in range(len(X_batch)):
            x = X_batch[j]
            X_batch_len.append(len(x))

            if max_len - len(x) > 0:
                pad = np.array([([0] * embedding_size)] * (max_len - len(x)))
                X_batch_res[j] = np.concatenate((x, pad))
            else:
                X_batch_res[j] = x

        y_batch, y_batch_len = converter.encode(y_batch)

        X_batch = torch.from_numpy(np.array(X_batch_res)).float()
        X_batch = X_batch.transpose(0,1)
        y_batch = torch.from_numpy(np.array(y_batch)).int()

        X_batch_len = torch.from_numpy(np.array(X_batch_len)).int()

        yield X_batch, y_batch, X_batch_len, y_batch_len

def accuracy(model: nn.Module, generator: Callable) -> float:
    """Returns the accuracy for a model.
    Args:
        model: A class inheriting from nn.Module.
        generator: A callable function returing a batch (data, labels).
    Returns:
        The accuracy for this model.
    """
    correct = 0
    total = 0

    for inputs, targets in generator():
        targets = targets.float()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = torch.round(torch.sigmoid(outputs))
        correct += (outputs == targets).count_nonzero().item() // len(targets[0])
        total += len(targets)
    return correct / total

def cer(model, converter, gen):
    d_sum = 0
    c_sum = 0

    for inputs, targets, inputs_len, targets_len in gen():

        out = model(inputs.to(device)).transpose(0,1)
        am = torch.argmax(out[:, :, :], 2)
        res = converter.decode(am, inputs_len)
        targets_dec = converter.decode(targets, targets_len)
        for i in range(len(targets)):
            d_sum += editdistance.eval(res[i], targets_dec[i])
            c_sum += len(targets_dec[i])
    cer = (100*d_sum/c_sum)
    return cer

def train_classifier(
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    n_epochs: int,
    train_gen: Callable,
    val_gen: Callable,
):
    """Train a model using a batch gradient descent.
    Args:
        model: a class inheriting from nn.Module.
        criterion: a loss criterion.
        optimizer: an optimizer
        n_epochs: the number of training epochs.
        train_gen: a callable function returing a batch (data, labels).
    """
    train_losses = np.zeros(n_epochs)
    train_accs = np.zeros(n_epochs)
    val_accs = np.zeros(n_epochs)

    for epoch in range(n_epochs):

        t0 = datetime.now()
        model.train()
        train_loss = []

        # Training loop.
        for inputs, targets in train_gen():

            # Put them on the GPU.
            inputs, targets = inputs.to(device), targets.to(device)

            # Reset the gradient.
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())  # .item() detach the value from GPU.

        train_losses[epoch] = np.mean(train_loss)
        model.eval()
        with torch.no_grad():
            train_accs[epoch] = accuracy(model, train_gen)
            val_accs[epoch] = accuracy(model, val_gen)
            scheduler.step(train_losses[epoch])

        print(f"Epoch: {epoch}, training loss: {train_losses[epoch]}, training accs: {train_accs[epoch]}, val accs: {val_accs[epoch]}, {datetime.now() - t0}")

    model.eval()
    return model, train_losses, train_accs, val_accs


def train_ctc(
    model: nn.Module,
    ctc: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    n_epochs: int,
    train_gen: Callable,
    val_gen: Callable,
    converter,
    evaluate_amount = 100
):
    """Train a model using a batch gradient descent.
    Args:
        model: a class inheriting from nn.Module.
        criterion: a loss criterion.
        optimizer: an optimizer
        n_epochs: the number of training epochs.
        train_gen: a callable function returing a batch (data, labels).
    """
    train_losses = np.zeros(n_epochs)
    train_cers = []
    val_cers = []

    sample_count = 0
    for epoch in range(n_epochs):

        t0 = datetime.now()
        model.train()
        train_loss = []

        # Training loop.
        for inputs, targets, inputs_len, targets_len in train_gen():
            
            # Put them on the GPU.
            inputs, targets, inputs_len, targets_len = inputs.to(device), targets.to(device), inputs_len.to(device), targets_len.to(device)
            sample_count += 1

            # Reset the gradient.
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = ctc(outputs.log_softmax(2), targets, inputs_len, targets_len)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())  # .item() detach the value from GPU.

            if sample_count > evaluate_amount:
                model.eval()
                with torch.no_grad():
                    train_cers.append(cer(model, converter, train_gen))
                    val_cers.append(cer(model, converter, val_gen))
                model.train()
                sample_count = 0

        train_losses[epoch] = np.mean(train_loss)
        model.eval()
        with torch.no_grad():
            scheduler.step(train_losses[epoch])

        print(f"Epoch: {epoch}, training loss: {train_losses[epoch]}, training cer: {train_cers[-1]}, val cer: {val_cers[-1]}, {datetime.now() - t0}")

    model.eval()
    return model, train_losses, train_cers, val_cers

