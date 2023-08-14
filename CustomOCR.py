import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import json
from CustomLSTM import CustomLSTMCTC
from pca2conv import initialize_conv2d
from tqdm import tqdm
import editdistance
from modules import Backbone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomOCR(torch.nn.Module):

    def __init__(self, nb_classes, feature_dim=200, lstm_layers=3):

        super().__init__()
        self.params = {'type': 'OCROnly', 'nb_classes': nb_classes, 'feature_dim': feature_dim, 'lstm_layers': lstm_layers}
        self.backbone = Backbone()

        self.rnn  = CustomLSTMCTC(self.backbone.output_dim, feature_dim, lstm_layers, nb_classes)

        self.act = torch.nn.ReLU()
        self.__length_map = {}
        self.__init_length_map()
    

    
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

    def initialize_backbone(self, batch, min_nb_crops):

        backbone = self.backbone
        
        batch = torch.nn.functional.pad(batch,(1,2,1,2))

        initialize_conv2d(backbone.conv1, batch, min_nb_crops)
        batch = backbone.act(backbone.conv1(batch))
        
        initialize_conv2d(backbone.conv2, batch, min_nb_crops)
        batch = backbone.max_pool1(backbone.act(backbone.conv2(batch)))

        initialize_conv2d(backbone.conv3, batch, min_nb_crops)
        batch = backbone.max_pool2(backbone.act(backbone.conv3(batch)))
        
        return batch

    def pca_weight_init(self, batch, min_nb_crops, pca_sample_size):
        batch = self.initialize_backbone(batch, min_nb_crops)

        batch = self.act(batch)
        batch = torch.mean(batch, axis=2)
        batch = batch.permute(2, 0, 1)

        batch = torch.flatten(batch, end_dim=1)

        self.rnn.pca_weight_init(batch,pca_sample_size)


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

        x = self.rnn(x)
        
        return x
    



def train_ocr(model, train_dataloader, valid_dataloader, ctc_loss, converter, optimizers, schedulers, filename):
    patience = 30
    cer=100

    with open(os.path.join(filename, 'logs.txt'), 'wt') as lfile:
        best_cer = 100
        no_imp = 0

        for epoch in range(1000):
            loss_sum = 0
            model.train()
            batches = 0
            for tns, base_width, lbl, base_length, _, _ in tqdm(train_dataloader, desc='%s, %d' % (filename, no_imp)):
                tns = tns.to(device)
                lbl = lbl.to(device)
                out = model(tns)
                il = model.convert_widths(base_width, out.shape[0])
                ol = torch.Tensor([l for l in base_length]).long()
                loss = ctc_loss(out.log_softmax(2), lbl, input_lengths=il, target_lengths=ol)
                for o in optimizers:
                    o.zero_grad()
                loss.backward()
                for o in optimizers:
                    o.step()
                loss_sum += loss.item()
                batches += 1
                
            with torch.no_grad():
                model.eval()
                d_sum = 0
                c_sum = 0
                for tns, base_width, lbl, lbl_width, _, _ in tqdm(valid_dataloader, position=0, leave=True, desc='Validation'):
                    out = model(tns.to(device)).transpose(0,1)
                    am = torch.argmax(out[:, :, :], 2)
                    res = converter.decode(am, base_width)
                    #lbl = converter.decode(lbl, lbl_width)
                    for i in range(len(lbl)):
                        d_sum += editdistance.eval(res[i], lbl[i])
                        c_sum += len(lbl[i])
                cer = (100*d_sum/c_sum)

                print('%d;%f;%f\n' % (epoch, (loss_sum/batches), cer))
                lfile.write('%d;%f;%f\n' % (epoch, loss_sum, cer))
                lfile.flush()
            model.train()

            for s in schedulers:
                s.step(cer)

            if cer<best_cer:
                no_imp = 0
                #model.save(filename)
                #model.save_optimizers(optimizers, filename)
                best_cer = cer
                #tqdm.write('Saved')
            else:
                no_imp += 1
            if no_imp>patience:
                tqdm.write('No improvement, lowest CER: %.2f' % best_cer)
                break
            tqdm.write('Loss sum: %.6f' % (loss_sum/batches))
            tqdm.write('     CER: %.2f' % cer)
