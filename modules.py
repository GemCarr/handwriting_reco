import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import json

class Backbone(torch.nn.Module):
    """
    CNN used in all of our OCR pipelines. 
    """ 
    def __init__(self, output_dim=64):
        """
        Constructor

        :param output_dim: number of neurons in the output layer
        :return: instance of the class
        """ 
        super().__init__()
        self.act = torch.nn.LeakyReLU()
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=(4,2))
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=(1,2))
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size= (3, 2), stride=(2,1))
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(6, 4), padding=(3,1))
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=output_dim, kernel_size=(3, 3), padding=(1,1))
        self.padding_params = 1/8
        self.output_dim = output_dim

    def forward(self, x):
        """
        Extracts features from an input text line

        :param x: text line (batch)
        :return: descriptors (batch)
        """ 
        x = torch.nn.functional.pad(x,(1,2,1,2))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.max_pool1(x)
        x = self.act(self.conv3(x))
        x = self.max_pool2(x)
        return x


class OCROnly(torch.nn.Module):
    """
    This is the base of all OCR models used in this project. It combines
    a CNN, an embedding layer, one or several LSTM layers, and an output
    layer.
    """
    def __init__(self, nb_classes, feature_dim=200, backbone=Backbone, lstm_layers=3):
        """
        Constructor of the class

        :param nb_classes: number of characters composing the alphabet
        :param feature_dim: number of neurons in the LSTM layers
        :param backbone: class of the backbone
        :param lstm_layers: self-describing parameter
        :return: an instance of the class
        """ 
        super().__init__()
        self.params = {'type': 'OCROnly', 'nb_classes': nb_classes, 'feature_dim': feature_dim, 'lstm_layers': lstm_layers}
        self.backbone = backbone()
        self.embed = torch.nn.Linear(self.backbone.output_dim, feature_dim)
        self.rnn  = torch.nn.LSTM(feature_dim, feature_dim, lstm_layers, bidirectional=True)
        self.head = torch.nn.Linear(2*feature_dim, nb_classes)
        self.act = torch.nn.ReLU()
        self.__length_map = {}
        self.__init_length_map()
    
    def load_from_folder(folder):
        """
        Static method loading an instance from a folder

        :param folder: source folder
        :return: an instance of the class
        """ 
        p = json.load(open(os.path.join(folder, 'params.json'), 'rt'))
        net = OCROnly(nb_classes=p['nb_classes'], feature_dim=p['feature_dim'], lstm_layers=p['lstm_layers'])
        net.load(folder)
        return net
    
    def save(self, folder):
        """
        Saves the model to a folder. Creates this folder if needed.

        :param folder: destination folder
        """ 
        os.makedirs(folder, exist_ok=True)
        torch.save(self.backbone.state_dict(), os.path.join(folder, 'backbone.pth'))
        torch.save(self.embed.state_dict(), os.path.join(folder, 'embed.pth'))
        torch.save(self.rnn.state_dict(), os.path.join(folder, 'rnn.pth'))
        torch.save(self.head.state_dict(), os.path.join(folder, 'head.pth'))
        json.dump(self.params, open(os.path.join(folder, 'params.json'), 'wt'), indent=4)
    
    def load(self, folder):
        """
        Loads the models' weights from a folder. Note that the model has
        to be properly initialized first.

        :param folder: source folder
        """ 
        self.backbone.load_state_dict(torch.load(os.path.join(folder, 'backbone.pth')))
        self.embed.load_state_dict(torch.load(os.path.join(folder, 'embed.pth')))
        self.rnn.load_state_dict(torch.load(os.path.join(folder, 'rnn.pth')))
        self.head.load_state_dict(torch.load(os.path.join(folder, 'head.pth')))
        self.params = json.load(open(os.path.join(folder, 'params.json'), 'rt'))
    
    def get_optimizers(self, folder=None):
        """
        Returns an array containing one optimizer - if folder is not none,
        then the optimizer's state dict stored in the folder is loaded.

        :param folder: source folder
        :return: an array containing one or optimizer
        """ 
        res = [
            torch.optim.Adam(self.parameters(), lr=0.001)
        ]
        if folder is not None:
            res[0].load_state_dict(torch.load(os.path.join(folder, 'optimizer.pth')))
        return res
    
    def save_optimizers(self, optimizers, folder):
        """
        Save the optimizer in a given folder.

        :param folder: destination folder
        """ 
        torch.save(optimizers[0].state_dict(), os.path.join(folder, 'optimizer.pth'))
    
    def convert_widths(self, w, max_width):
        """
        Converts an input widths (in pixel columns) to output widths (in
        the output tensor). Returned as tensor.

        :param w: tensor or array containing a list of width
        :return: long tensor containing the converted widths
        """ 
        return torch.Tensor([min(self.__length_map[x], max_width) for x in w]).long()
    
    def __init_length_map(self):
        """
        Initializes the map conversion system for convert_width(). Note
        that it tries to cache the resuts in dat/length_map.json.
        """ 
        max_length = 2500
        try:
            self.__length_map = json.load(open(os.path.join('dat', 'length_map.json'), 'rt'))
            return
        except: pass
        
        tns = torch.zeros(1, 1, 8, max_length)
        with torch.no_grad():
            out  = self.backbone(tns)
            last = out[0][0][0][out.shape[3]//2]
            ls  = 0
            pos = 0
            self.__length_map = []
            for i in range(max_length):
                tns[0,0,:,i] = i
                out = torch.sum(self.backbone(tns), axis=1)
                while pos<out.shape[2]-1 and out[0,0,pos]!=out[0,0,pos+1]:
                    pos += 1
                self.__length_map.append(pos-1)
        try:
            json.dump(self.__length_map, open(os.path.join('dat', 'length_map.json'), 'wt'))
        except: pass
    
    def forward(self, x):
        """
        Processes an input batch.

        :param x: input batch
        :return: the network's output, ready to be convered to a string
        """ 
        x = self.backbone(x)
        x = self.act(x)
        x = torch.mean(x, axis=2)
        x = x.permute(2, 0, 1)
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = self.head(x)
        return x

