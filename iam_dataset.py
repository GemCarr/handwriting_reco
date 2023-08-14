import json
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pickle
from converter import split

cs = json.load(open(os.path.join('ocr', 'iam_charmap.json'), 'rt'))
cs['PAD'] = len(cs)+1

no_font = 0

def iam_collate_batch(batch):
    """
    Creates a batch for PyTorch, out of the data loaded by IAMDataset

    :param batch: input batch
    :return: image batch, image lengths, transcriptions, transcription lengths
    """ 
    im_list = []
    txt_list = []
    for (im, txt, cf, pf) in batch:
        im_list.append(im.permute(2, 0, 1))
        txt_list.append(txt)
    base_im_width = [im.shape[0] for im in im_list]
    base_txt_length = [len(txt) for txt in txt_list]
    im_list = pad_sequence(im_list, batch_first=False, padding_value=0).permute(1, 2, 3, 0)
    im_list = torch.stack([x for x in im_list])
    
    if type(txt)!=str:
        txt_list = pad_sequence(txt_list, batch_first=True, padding_value=cs['PAD'])
    return im_list, base_im_width, txt_list, base_txt_length, None, None

class IAMDataset(Dataset):
    """
    This is the main class

    :param batch: input batch
    :return: image batch, image lengths, transcriptions, transcription lengths
    """ 
    def __init__(self, folder, charset, transform=None, height=32, cache=True):
        """
        Constructor of the class. The parameter folder can be either a string
        pointing to the folder containing the data, or an array containing
        multiple paths.

        :param folder: folder or list of folders containing the data
        :param charset: mapping from characters to their corresponding class numbers
        :param transform: augmentation transforms to apply
        :param height: to which text lines are resized
        :param cache: keep non-augmented data in the RAM
        :return: instance of the class
        """ 
        if type(folder)==str:
            self.base_names = [os.path.join(folder, x.replace('.txt', '')) for x in os.listdir(folder) if x.endswith('.txt')]
        else: # assuming it's a tuple
            self.base_names = []
            for fld in folder:
                self.base_names += [os.path.join(fld, x.replace('.txt', '')) for x in os.listdir(fld) if x.endswith('.txt')]
        self.transform = transform
        self.height = height
        self.charset = charset
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            im, text = self.cache[idx]
            if self.transform:
                im = self.transform(im)
            return im, text, None, None
        im = Image.open('%s.png' % self.base_names[idx]).convert('RGB')
        if im.size[1]!=self.height:
            ratio = self.height / im.size[1]
            width = int(im.size[0] * ratio)
            try:
                im = im.resize((width,self.height), Image.Resampling.LANCZOS)
            except:
                print('Cannot resize', self.base_names[idx])
                quit(1)
        if self.charset is not None:
            try:
                text = torch.Tensor([self.charset[x] for x in split(open('%s.txt' % self.base_names[idx]).read().strip())])
            except:
                print('Failed to read')
                print('%s.txt' % self.base_names[idx])
                quit(1)
        else:
            text = open('%s.txt' % self.base_names[idx]).read().strip()
        
        if self.cache is not None:
            self.cache[idx] = (im, text)
        if self.transform:
            im = self.transform(im)
        return im, text, None, None
