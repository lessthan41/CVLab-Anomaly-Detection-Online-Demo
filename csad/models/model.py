import torch.nn as nn
import torch
import torch.nn.functional as F
import timm
from models.model_utils import make_layer,BasicBlock, Bottleneck
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# print(summary(wide_resnet101_2().cuda(), (3, 512, 512)))


def positionalencoding2d(D, H, W):
    """
    taken from https://github.com/gudovskiy/cflow-ad
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(np.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P.cuda()[None]

class PDN_S(nn.Module):

    def __init__(self, last_kernel_size=384,padding=False):
        super(PDN_S, self).__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 128 3 ReLU
        # AvgPool-1 2×2 2×2 128 1 -
        # Conv-2 1×1 4×4 256 3 ReLU
        # AvgPool-2 2×2 2×2 256 1 -
        # Conv-3 1×1 3×3 256 1 ReLU
        # Conv-4 1×1 4×4 384 0 -
        self.padding = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1*self.padding)
        self.conv4 = nn.Conv2d(256, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        return x
    
class PDN_M(nn.Module):

    def __init__(self, last_kernel_size=384,padding=False,stae=True):
        super(PDN_M, self).__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 256 3 ReLU
        # AvgPool-1 2×2 2×2 256 1 -
        # Conv-2 1×1 4×4 512 3 ReLU
        # AvgPool-2 2×2 2×2 512 1 -
        # Conv-3 1×1 1×1 512 0 ReLU
        # Conv-4 1×1 3×3 512 1 ReLU
        # Conv-5 1×1 4×4 384 0 ReLU
        # Conv-6 1×1 1×1 384 0 -
        self.stae = stae
        self.last_kernel_size = last_kernel_size
        self.padding = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1*self.padding)
        self.conv5 = nn.Conv2d(512, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(last_kernel_size, last_kernel_size, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        if self.stae:
            return x[:,:self.last_kernel_size//2,:,:],x[:,self.last_kernel_size//2:,:,:]
        else:
            return x
        
class PDN_M_Multi(nn.Module):

    def __init__(self, out_dim=512,feat_size=80,padding=False):
        super(PDN_M_Multi, self).__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 256 3 ReLU
        # AvgPool-1 2×2 2×2 256 1 -
        # Conv-2 1×1 4×4 512 3 ReLU
        # AvgPool-2 2×2 2×2 512 1 -
        # Conv-3 1×1 1×1 512 0 ReLU
        # Conv-4 1×1 3×3 512 1 ReLU
        # Conv-5 1×1 4×4 384 0 ReLU
        # Conv-6 1×1 1×1 384 0 -
        self.out_dim = out_dim
        self.st_size = feat_size
        self.ae_size = feat_size
        self.padding = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1*self.padding)
        self.conv5 = nn.Conv2d(512, self.out_dim, kernel_size=4, stride=1, padding=0)
        self.conv_st = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        self.conv_ae = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        st = self.conv_st(x)
        st = F.interpolate(st, size=self.st_size, mode='bilinear', align_corners=False)
        ae = self.conv_ae(x)
        ae = F.interpolate(ae, size=self.ae_size, mode='bilinear', align_corners=False)
        return st,ae
        
class EncConv(nn.Module):

    def __init__(self):
        super(EncConv, self).__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # EncConv-1 2×2 4×4 32 1 ReLU
        # EncConv-2 2×2 4×4 32 1 ReLU
        # EncConv-3 2×2 4×4 64 1 ReLU
        # EncConv-4 2×2 4×4 64 1 ReLU
        # EncConv-5 2×2 4×4 64 1 ReLU
        # EncConv-6 1×1 8×8 64 0 -
        self.enconv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)
        # self.apply(weights_init)

    def forward(self, x):
        # pdb.set_trace()
        x1 = F.relu(self.enconv1(x))
        #x = x + positionalencoding2d(32, 128, 128)
        x2 = F.relu(self.enconv2(x1))
        x3 = F.relu(self.enconv3(x2))
        x4 = F.relu(self.enconv4(x3))
        x5 = F.relu(self.enconv5(x4))
        x6 = self.enconv6(x5)
        return x6

class DecConv(nn.Module):

    def __init__(self,padding=False):
        super(DecConv, self).__init__()
        # Bilinear-1 Resizes the 1×1 input features maps to 3×3
        # DecConv-1 1×1 4×4 64 2 ReLU
        # Dropout-1 Dropout rate = 0.2
        # Bilinear-2 Resizes the 4×4 input features maps to 8×8
        # DecConv-2 1×1 4×4 64 2 ReLU
        # Dropout-2 Dropout rate = 0.2
        # Bilinear-3 Resizes the 9×9 input features maps to 15×15
        # DecConv-3 1×1 4×4 64 2 ReLU
        # Dropout-3 Dropout rate = 0.2
        # Bilinear-4 Resizes the 16×16 input features maps to 32×32
        # DecConv-4 1×1 4×4 64 2 ReLU
        # Dropout-4 Dropout rate = 0.2
        # Bilinear-5 Resizes the 33×33 input features maps to 63×63
        # DecConv-5 1×1 4×4 64 2 ReLU
        # Dropout-5 Dropout rate = 0.2
        # Bilinear-6 Resizes the 64×64 input features maps to 127×127
        # DecConv-6 1×1 4×4 64 2 ReLU
        # Dropout-6 Dropout rate = 0.2
        # Bilinear-7 Resizes the 128×128 input features maps to 64×64
        # DecConv-7 1×1 3×3 64 1 ReLU
        # DecConv-8 1×1 3×3 384 1 -
        self.padding = padding

        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)
        # self.dropout1 = nn.Identity()
        # self.dropout2 = nn.Identity()
        # self.dropout3 = nn.Identity()
        # self.dropout4 = nn.Identity()
        # self.dropout5 = nn.Identity()
        # self.dropout6 = nn.Identity()
        # self.apply(weights_init)


    def forward(self, x):
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=8, mode='bilinear')
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=15, mode='bilinear')
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=32, mode='bilinear')
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=63, mode='bilinear')
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=127, mode='bilinear')
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        if self.padding:
            x = F.interpolate(x, size=64, mode='bilinear')
        else:
            x = F.interpolate(x, size=56, mode='bilinear') #64
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.in_channels != self.out_channels:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Compensation(nn.Module):
    def __init__(self, in_channels,in_shape,input_embed,template_embed,out_channels=None,only_relation=False):
        super(Compensation, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.only_relation = only_relation
        if only_relation:
            self.final_conv = nn.Sequential(
                nn.Conv2d(in_shape[0]*in_shape[1]*2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            )
            
        else:
            self.final_conv =  nn.Sequential(
                nn.Conv2d(input_embed+template_embed+in_shape[0]*in_shape[1]*2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            )
        self.input_embed = nn.Conv2d(in_channels, input_embed, kernel_size=1, stride=1, padding=0, bias=False)
        self.template_embed = nn.Conv2d(in_channels, template_embed, kernel_size=1, stride=1, padding=0, bias=False)
    
    def correlation(self, x, template_x):
        C,H,W = x.shape
        flat_x = x.view(C,H*W) # CxHxW -> Cx(HxW)
        flat_x = flat_x.transpose(0,1) # Cx(HxW) -> (HxW)xC
        flat_template_x = template_x.view(C,H*W) # CxHxW -> Cx(HxW)
        result_vec = torch.matmul(flat_x,flat_template_x) # (HxW)xC x Cx(HxW) -> (HxW)x(HxW)
        result_vec = result_vec.transpose(0,1) # (HxW)x(HxW) -> (HxW)x(HxW) to channel first
        result_vec = result_vec.view(H*W,H,W) # (HxW)x(HxW) -> (HxW)xHxW
        return result_vec
    
    def correlation2(self, x, template_x):
        B, C, H, W = x.shape

        flat_x = x.view(B, C, H * W).transpose(1, 2)  # BxCxHxW -> Bx(HxW)xC
        flat_template_x = template_x.view(B, C, H * W)  # BxCxHxW -> BxCx(HxW)

        result_vec = torch.matmul(flat_x, flat_template_x)  # Bx(HxW)xC x BxCx(HxW) -> Bx(HxW)x(HxW)
        result_vec = result_vec.transpose(1, 2)  # Bx(HxW)x(HxW) -> Bx(HxW)x(HxW) to channel first
        result_vec = result_vec.view(B, H * W, H, W)  # Bx(HxW)x(HxW) -> Bx(HxW)xHxW

        return result_vec
    
    def relation_tensor(self, x, template_x):
        # x: Bx64xHxW
        # template_x: Bx64xHxW
        result = []
        for i in range(x.shape[0]): # B
            relation = torch.concat([self.correlation(x[i],template_x[i]),self.correlation(template_x[i],x[i])],dim=0) # (2xHxW)xHxW
            result.append(relation)
        return torch.stack(result,dim=0) # Bx(2xHxW)xHxW
    
    def relation_tensor2(self, x, template_x):
        # x: Bx64xHxW
        # template_x: Bx64xHxW
        # channel wise concat
        result = torch.concat([self.correlation2(x, template_x),self.correlation2(template_x, x)],dim=1) # Bx(2xHxW)xHxW
        return result # Bx(2xHxW)xHxW
    
    def forward(self, x, template_x):
        # x: Bx64xHxW
        # template_x: Bx64xHxW
        x = self.input_embed(x)
        template_x = self.template_embed(template_x)
        if self.only_relation:
            x = self.relation_tensor2(x,template_x)
            x = self.final_conv(x)
            return x
        else:
            x = torch.concat([x,template_x,self.relation_tensor2(x,template_x)],dim=1)
            x = self.final_conv(x)
            return x
        
class AutoEncoder(nn.Module):
    def __init__(self, out_size=64,out_dim=384,base_dim=64,input_size=320):
        super(AutoEncoder, self).__init__()
        self.out_dim = out_dim
        self.base_dim = base_dim
        self.input_size = input_size
        self.out_size = out_size

        self.enconv1 = nn.Conv2d(3, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=8, stride=1, padding=0)

        self.deconv1 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(self.base_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)


    def forward(self, x):
        x1 = F.relu(self.enconv1(x))#128
        # x1 = F.adaptive_avg_pool2d(x1, (64, 64))
        x2 = F.relu(self.enconv2(x1))#64
        # x2 = F.adaptive_avg_pool2d(x2, (32, 32))
        x3 = F.relu(self.enconv3(x2))#32
        # x3 = F.adaptive_avg_pool2d(x3, (16, 16))
        x4 = F.relu(self.enconv4(x3))#16
        # x4 = F.adaptive_avg_pool2d(x4, (8, 8))
        x5 = F.relu(self.enconv5(x4))#8
        x6 = self.enconv6(x5)
        # x6 = F.adaptive_avg_pool2d(x5, (1, 1))

        x = F.interpolate(x6, size=3, mode='bilinear')
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=8, mode='bilinear')
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=15, mode='bilinear')
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=32, mode='bilinear')
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=63, mode='bilinear')
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=127, mode='bilinear')
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=self.out_size, mode='bilinear')
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x
    
class AutoEncoderSegmap(nn.Module):
    def __init__(self,segmentor=None,out_feat_dim=384,embed_dim=32,out_size=80,pad2resize=None):
        super(AutoEncoderSegmap, self).__init__()
        # load segmentor
        self.embed_dim = embed_dim
        self.out_size = out_size
        self.base_dim = 64
        self.out_feat_dim = out_feat_dim
        
        self.segmentor = segmentor
        self.pad2resize = pad2resize
        self.num_classes = self.segmentor.fc2.conv3.out_channels
        
        self.out_dim = self.out_feat_dim + self.num_classes
        self.seg_embedding = nn.Conv2d(self.num_classes, int(self.embed_dim//2), kernel_size=1, stride=1, padding=0)
        self.image_embedding = nn.Conv2d(3, int(embed_dim//2), kernel_size=1, stride=1, padding=0)

        self.enconv1 = nn.Conv2d(self.embed_dim, int(self.base_dim//2), kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(int(self.base_dim//2), int(self.base_dim//2), kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(int(self.base_dim//2), self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=8, stride=1, padding=0)

        self.deconv1 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(self.base_dim, self.out_dim, kernel_size=3, stride=1, padding=1)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)


    def forward(self, x):
        input_size = x.size()[2:]
        with torch.no_grad():
            self.segmentor.eval()
            segmap = self.segmentor(x)
            segmap = torch.softmax(segmap,dim=1)
            segmap = self.pad2resize(segmap)
        seg_embed = self.seg_embedding(segmap)
        image_embed = self.image_embedding(x)
        
        concat_embed = torch.concat([image_embed,seg_embed],dim=1)
        x = concat_embed
            
        x1 = F.relu(self.enconv1(x))
        x2 = F.relu(self.enconv2(x1))
        x3 = F.relu(self.enconv3(x2))
        x4 = F.relu(self.enconv4(x3))
        x5 = F.relu(self.enconv5(x4))
        x6 = self.enconv6(x5)

        x = F.interpolate(x6, size=3, mode='bilinear')
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=8, mode='bilinear')
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=15, mode='bilinear')
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=31, mode='bilinear')
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=63, mode='bilinear')
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=127, mode='bilinear')
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=self.out_size, mode='bilinear')
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        
        ae_out = x[:,:self.out_feat_dim]
        seg_rec = F.interpolate(x[:,self.out_feat_dim:], size=input_size,mode='bilinear')
        return ae_out, seg_rec, segmap

class AutoEncoderTemplate(nn.Module):
    def __init__(self, padding=False):
        super(AutoEncoderTemplate, self).__init__()
        self.encoder = EncConv()
        #self.decoder = DecConv(padding)
        self.bottle_neck = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.compensation1 = Compensation(in_channels=64,in_shape=[16,16],
                                          input_embed=64,template_embed=1,out_channels=64)
        self.compensation2 = Compensation(in_channels=64,in_shape=[32,32],
                                          input_embed=1,template_embed=64,out_channels=64)
        self.compensation3 = Compensation(in_channels=64,in_shape=[64,64],
                                          input_embed=1,template_embed=64,out_channels=64)

        self.padding = padding

        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=1)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)
        #self.linear = nn.Linear(512, 64)

    def forward(self, x,clip_features=None,template=None):
        x2,x3,x4, x5,x6 = self.encoder(x)
        t2,t3,t4, t5,t6 = self.encoder(template)

        x = F.interpolate(x6, size=3, mode='bilinear')
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=8, mode='bilinear')
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)



        x = F.interpolate(x, size=15, mode='bilinear')
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)

        ###################### 
        x = self.compensation1(x,t4)

        x = F.interpolate(x, size=32, mode='bilinear')
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)

        #####################
        x = self.compensation2(x,t3)

        x = F.interpolate(x, size=63, mode='bilinear')
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)

        ####################
        x = self.compensation3(x,t2)

        x = F.interpolate(x, size=127, mode='bilinear')
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        if self.padding:
            x = F.interpolate(x, size=64, mode='bilinear')
        else:
            x = F.interpolate(x, size=56, mode='bilinear') #64
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x
    

class AutoEncoderResNet(nn.Module):
    def __init__(self,encoder_type='resnet18',bottleneck_type='template'):
        super().__init__()
        if encoder_type == 'resnet18':
            self.encoder = timm.create_model('resnet18', pretrained=False,
                                             features_only=True, out_indices=[1, 2, 3, 4])
            bottleneck_dim = 512
            # [64,64,64]
            # [128,32,32]
            # [256,16,16]
            # [512,8,8]
        elif encoder_type == 'wideresnet50_2':
            self.encoder = timm.create_model('wide_resnet50_2.tv2_in1k', pretrained=False,
                                             features_only=True, out_indices=[1, 2, 3, 4])
            bottleneck_dim = 2048
            # [256,64,64]
            # [512,32,32]
            # [1024,16,16]
            # [2048,8,8]
        if bottleneck_type == 'resnet':
            self.bottleneck = Bottleneck(bottleneck_dim,bottleneck_dim)
        elif bottleneck_type == 'basic':
            self.bottleneck = BasicBlock(bottleneck_dim,bottleneck_dim)
        elif bottleneck_type == 'template':
            self.bottleneck = nn.Sequential(
                BasicBlock(bottleneck_dim,bottleneck_dim),
                #nn.MaxPool2d(kernel_size=8),
                nn.AvgPool2d(kernel_size=8),
                nn.ConvTranspose2d(bottleneck_dim, bottleneck_dim, kernel_size=2, stride=2, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(bottleneck_dim, bottleneck_dim, kernel_size=2, stride=2, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(bottleneck_dim, bottleneck_dim, kernel_size=2, stride=2, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )



        self.stage1 = make_layer(BasicBlock, bottleneck_dim, bottleneck_dim, 3)
        self.stage2 = make_layer(BasicBlock, bottleneck_dim, bottleneck_dim//2, 4)
        self.stage3 = make_layer(BasicBlock, bottleneck_dim//2, bottleneck_dim//4, 6)
        self.stage4 = make_layer(BasicBlock, bottleneck_dim//4, bottleneck_dim//8, 3)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x):
        _, _, _, x = self.encoder(x)

        x = self.bottleneck(x)
        
        x3 = self.stage1(x)
        x2 = self.stage2(F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False))
        x1 = self.stage3(F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False))
        x0 = self.stage4(F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False))

        print(x0.shape,x1.shape,x2.shape)
        return [x0, x1, x2]


class Teacher(nn.Module):
    def __init__(self,size,channel_size=384, padding=False):
        super(Teacher, self).__init__()
        if size =='M':
            self.pdn = PDN_M(last_kernel_size=channel_size,padding=padding)
        elif size =='S':
            self.pdn = PDN_S(last_kernel_size=channel_size,padding=padding)
        # self.pdn.apply(weights_init)

    def forward(self, x):
        x = self.pdn(x)
        return x
    
class ResNetTeacher(nn.Module):
    def __init__(self,out_dim=256,feat_size=64):
        super(ResNetTeacher, self).__init__()
        import timm
        self.encoder = timm.create_model('wide_resnet50_2'
                                          ,pretrained=True,
                                          features_only=True,
                                          out_indices=[2,3])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.out_dim = out_dim
        self.feat_size = feat_size
        self.proj = nn.Conv2d(1024+512, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.proj.requires_grad_(False)
        
    def forward(self, x):
        x = self.encoder(x)
        concat_feat = []
        for i in range(1,len(x)):
            feat = x[i]
            feat = F.interpolate(feat, size=self.feat_size, mode='bilinear',align_corners=False)
            concat_feat.append(feat)
        concat_feat = torch.cat(concat_feat,dim=1)
        concat_feat = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(concat_feat)
        proj_feat = self.proj(concat_feat)
        return proj_feat
    

class Student(nn.Module):
    def __init__(self,size,channel_size=384, padding=False) -> None:
        super(Student, self).__init__()
        if size =='M':
            self.pdn = PDN_M(last_kernel_size=channel_size*2,padding=padding) #The student network has the same architecture,but 768 kernels instead of 384 in the Conv-5 and Conv-6 layers.
        elif size =='S':
            self.pdn = PDN_S(last_kernel_size=channel_size*2,padding=padding) #The student network has the same architecture, but 768 kernels instead of 384 in the Conv-4 layer
        # self.pdn.apply(weights_init)

    def forward(self, x):
        pdn_out = self.pdn(x)
        return pdn_out
    

class AutoEncoderOrig(nn.Module):
    def __init__(self, padding=False):
        super(AutoEncoderOrig, self).__init__()
        self.encoder = EncConv()
        self.decoder = DecConv(padding)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    import torch
    import time
    from PIL import Image
    import torchsummary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AutoEncoder(
        out_size=64,
        out_dim=512,
        base_dim=64,
        input_size=256
    ).cuda()
    # model(torch.randn(4,3,256,256).cuda())
    model2 = AutoEncoderOrig(padding=True).cuda()
    torchsummary.summary(model, (3, 256, 256))
    torchsummary.summary(model2, (3, 256, 256))



    # correlation = Compensation(in_channels=64,in_shape=[16,16]).cuda()
    # x = torch.randn(8,64,16,16).cuda()
    # template_x = torch.randn(8,64,16,16).cuda()
    # out = correlation(x,template_x)
    # print(out.shape)

    # result = correlation.relation_tensor(x,template_x)
    # result2 = correlation.relation_tensor2(x,template_x)
    # print(torch.max(torch.abs(result-result2)))
    # print(torch.allclose(result,result2))
    # result = result.cpu().detach().numpy()
    # result2 = result2.cpu().detach().numpy()
    # print(np.where((result-result2)>0.0001))
    
    # t = time.time()
    # for i in range(5000):
    #     x = torch.randn(8,64,16,16).cuda()
    #     template_x = torch.randn(8,64,16,16).cuda()
    #     result = correlation.relation_tensor2(x,template_x)
    # print(time.time()-t)
    # t = time.time()
    # for i in range(5000):
    #     x = torch.randn(8,64,16,16).cuda()
    #     template_x = torch.randn(8,64,16,16).cuda()
    #     result = correlation.relation_tensor(x,template_x)
    # print(time.time()-t)
    

    # a = open_clip.list_pretrained()
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    # image = preprocess(Image.open("C:/Users/kev30/Desktop/test/code/anomaly/effcientAD/EfficientAD/models/000.png")).unsqueeze(0)
    # with torch.no_grad():
    #     image_features = model.encode_image(image)

    
    
    #model = AutoEncoder()
    #summary(model, (3, 256, 256))
    
 
    