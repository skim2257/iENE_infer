from torch import nn
from torchvision.models import resnet18 #, resnet34#, resnet50, resnet101#, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,\
                               #densenet121, densenet169, densenet161, densenet201
from acsconv.converters import ACSConverter

class ACSModel(nn.Module):
    """
    ACS Convolution nn.Module. 
    
    References
    ... [1] https://arxiv.org/abs/1911.10477
    """
    def __init__(self,
                 resnet: str = "resnet18",
                 pretrained: bool = True,
                 act: str = 'relu',
                 dropout: float = 0.3,
                 num_tasks: int = 1):
        """
        Parameters
        ----------
        arch
            Name of published model architecture
        pretrained
            Whether to load ImageNet pretrained weights or not
        act
            Activation function
        dropout
            Amount of dropout in fc layers. [0, 1]
        """
        super(ACSModel, self).__init__()
        
        activations = {'relu': nn.ReLU,
                       'prelu': nn.PReLU,
                       'leakyrelu': nn.LeakyReLU, 
                       'selu': nn.SELU,
                       'elu': nn.ELU}
        
        self.model_3d = ACSConverter(resnet18()).cuda()
        self.activation = activations[act]
        
        self.model_3d.fc = nn.Sequential(
            self.activation(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            self.activation(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_tasks),
            nn.Sigmoid()).cuda()
        
    def forward(self, x):
        """
        Forward prop on samples
        """
        return self.model_3d(x)