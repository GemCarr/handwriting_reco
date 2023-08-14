import torch
import torchvision.models as models
from torchvision import transforms

def __make_batch(im, size=128, n=128):
    """
    This function, used for debugging purpose only, creates a batch
    out of an input PIL image.

    :param im: PIL image
    :param size: size of the crops
    :param n: how many crops have to be extracted
    :return: a batch tensor
    """ 
    b   = []
    rc  = transforms.RandomCrop(size)
    tot = transforms.ToTensor()
    for i in range(n):
        b.append(tot(rc(im)))
    return torch.stack(b)

def __make_crops(batch, dim, min_nb_crops):
    """
    This private function extracts random crops if specified dimensions
    from a batch. This is used for creating samples to produce the
    PCA. Note that the number of crops extracted corresponds to the
    smallest multiple of the batch size which is greater than
    min_nb_crops.

    :param batch: input batch
    :param size: size of the crops (input size of conv2d)
    :param min_nb_crops: lower bound of the amount of crops to extract
    :return: a tensor stacking all crops
    """ 
    cropper = transforms.RandomCrop(dim)
    res = []
    while len(res)<min_nb_crops:
        for i in range(batch.shape[0]):
            res.append(cropper(batch[i]))
    return torch.stack(res)

def initialize_conv2d(conv, batch, min_nb_crops):
    """
    This method initializes a conv2d with PCA using the data from
    a batch. Sets the weights of the conv2d using the projection matrix,
    and the bias (if there are any) to center the data. Note that the
    matrix is normalized such that the projected data has unit variance
    on each axis.
    The PCA is computed using crops from the batch. The number of used
    crops is the smallest multiple of the batch size greater than
    min_nb_crops.
    Note that if the PCA cannot produce enough channels, then multiple
    PCA will be computed on different sets of crops.

    :param conv: the conv2d to initialize
    :param batch: suitable input batch for the conv2d
    :param min_nb_crops: lower bound of the amount of crops to extract
    :return: nothing
    """ 
    kernel_size   = conv.weight.shape[2:]
    output_dim    = conv.weight.shape[0]
    reshape_shape = [-1] + [x for x in conv.weight.shape[1:]]
    with torch.no_grad():
        new_weights = None
        new_bias    = None
        while new_weights is None or new_weights.shape[0]<output_dim:
            crops = __make_crops(batch, dim=kernel_size, min_nb_crops=min_nb_crops)
            crops = crops.view(crops.shape[0], -1)
            step_size = min(crops.shape)
            _, _, v = torch.pca_lowrank(crops, q=step_size, center=True, niter=2)
            std = torch.std(torch.matmul(crops-torch.mean(crops, axis=0), v), dim=0, unbiased=True)
            v = v / std
            if new_weights is None:
                new_weights = v.transpose(0, 1).reshape(reshape_shape)
            else:
                new_weights = torch.concat((new_weights, v.transpose(0, 1).reshape(reshape_shape)))
            if conv.bias is not None:
                if new_bias is None:
                    new_bias = -torch.matmul(torch.mean(crops, axis=0), v)
                else:
                    new_bias = torch.concat((new_bias, -torch.matmul(torch.mean(crops, axis=0), v)))
        new_weights = new_weights[:output_dim, :, :, :]
        if conv.bias is not None:
            new_bias  = new_bias[:output_dim]
            conv.bias = torch.nn.Parameter(new_bias)
        conv.weight = torch.nn.Parameter(new_weights)

def initialize_resnet(net, batch, min_nb_crops, debug=False):
    """
    This method initializes a resnet (official torchvision
    implementation required) using PCA. The different layers of the
    network are initialized sequentially.
    For each convolution, a number of crops will be extracted to
    to compute the PCA. While the batch should be as large as possible,
    some memory must be kept available for these crops.

    :param net: the network to initialize (must be an instance of a
                torchvision resnet)
    :param batch: suitable input batch for the resnet
    :param min_nb_crops: lower bound of the amount of crops to extract
    :return: nothing
    """ 
    
    if debug: print('Creating', type(net))
    
    if debug: print('Initializing conv1')
    initialize_conv2d(net.conv1, batch, min_nb_crops)
    
    batch = net.maxpool(net.relu(net.bn1(net.conv1(batch))))
    
    for l_num, layer in enumerate((net.layer1, net.layer2, net.layer3, net.layer4)):
        if debug: print('Initializing layer%d' % (l_num+1))
        for block in layer:
            if type(block)==models.resnet.BasicBlock:
                if debug: print(' Initializing BasicBlock')
                identity = batch
                if debug: print('  Initializing conv1')
                initialize_conv2d(block.conv1, batch, min_nb_crops)
                batch = block.relu(block.bn1(block.conv1(batch)))
                if debug: print('  Initializing conv2')
                initialize_conv2d(block.conv2, batch, min_nb_crops)
                batch = block.bn2(block.conv2(batch))
                if block.downsample is not None:
                    if debug: print('  Initializing downsampler')
                    try:
                        initialize_conv2d(block.downsample[0], identity, min_nb_crops)
                    except: pass # TODO : implement what to do if the layer is too big
                    identity = block.downsample(identity)
                batch = block.relu(batch+identity)
            elif type(block)==models.resnet.Bottleneck:
                if debug: print(' Initializing Bottleneck')
                identity = batch
                if debug: print('  Initializing conv1')
                initialize_conv2d(block.conv1, batch, min_nb_crops)
                batch = block.relu(block.bn1(block.conv1(batch)))
                if debug: print('  Initializing conv2')
                initialize_conv2d(block.conv2, batch, min_nb_crops)
                batch = block.relu(block.bn2(block.conv2(batch)))
                if debug: print('  Initializing conv3')
                initialize_conv2d(block.conv3, batch, min_nb_crops)
                batch = block.bn3(block.conv3(batch))
                if block.downsample is not None:
                    if debug: print('  Initializing downsampler')
                    initialize_conv2d(block.downsample[0], identity, min_nb_crops)
                    identity = block.downsample(identity)
                batch = block.relu(batch+identity)
    return net

if __name__ == "__main__":
    from PIL import Image
    
    print('Creating batch')
    # Note that any batch is fine, it does not need to be created in this
    # way. Just use a batch from a data loader.
    batch = __make_batch(Image.open('sample.jpg').convert('RGB'), n=512)
    
    print('Creating network')
    net = models.resnet18()
    net.conv1.bias = torch.nn.Parameter(torch.Tensor([1]))
    net = initialize_resnet(net, batch, 64, debug=True)
    print(net.conv1.bias.shape)
