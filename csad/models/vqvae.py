import torch
from torch import nn
from torch.nn import functional as F

import models.distributed as dist_fn

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).cuda()
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return  z_q, loss, perplexity, min_encodings, min_encoding_indices

class VectorQuantizer2(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta=0.25):
        super(VectorQuantizer2, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        


        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        # add Gumbel noise
        if self.training:
            # generate random binary mask
            mask = torch.randint(0, 2, size=z_flattened.shape).cuda()
            # get indices of nonzero elements
            mask_indices = torch.nonzero(mask)
            # get indices of zero elements
            mask_indices_zero = torch.nonzero(1 - mask)

            g = np.random.gumbel(loc=0,scale=1,size=d.shape)
            g = torch.from_numpy(g).cuda()*(torch.mean(torch.abs(d))*0.5)
            d = d + g

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).cuda()
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return  z_q, loss, perplexity, min_encodings, min_encoding_indices

    
class VQVAE(nn.Module):
    def __init__(self,is_bn=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = EncConv()
        self.decoder = DecConv(is_bn=is_bn)
        self.quantizer = VectorQuantizer2(n_e=16, e_dim=64, beta=0.25)

    def forward(self, x):
        x = self.encoder(x)
        x, vq_loss, _, _, _ = self.quantizer(x)
        x = self.decoder(x)
        return x, vq_loss

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out
    
class VQVAE2(nn.Module):
    def __init__(self,embed_dim=64, top_dim=64,bottom_dim=128,out_dim=384, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # input 256x256x3 -> 64x64x32
        # top encoder 64x64x32 -> 16x16x64
        # bottom encoder 16x16x64 -> 4x4x128
        # bottom decoder 4x4x128 -> 16x16x64
        # top decoder 16x16x64 -> 64x64x384
        self.top_dim = top_dim
        self.bottom_dim = bottom_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        
        
        self.head = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),# 128x128x32
                                    ResBlock(32, 32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),# 64x64x32
                                    ResBlock(32, 32),
                                    nn.ReLU(),)
        self.encoder_top = nn.Sequential(nn.Conv2d(32, self.top_dim, kernel_size=4, stride=2, padding=1),# 32x32x64
                                            ResBlock(self.top_dim, self.top_dim),
                                            nn.ReLU(),
                                            nn.Conv2d(self.top_dim,self.top_dim , kernel_size=4, stride=2, padding=1),# 16x16x64
                                            ResBlock(self.top_dim, self.top_dim),
                                            nn.ReLU(),)
        self.encoder_bottom = nn.Sequential(nn.Conv2d(self.top_dim, self.bottom_dim, kernel_size=4, stride=2, padding=1),# 8x8x128 
                                            ResBlock(self.bottom_dim, self.bottom_dim),
                                            nn.ReLU(),
                                            nn.Conv2d(self.bottom_dim, self.bottom_dim, kernel_size=4, stride=2, padding=1),# 4x4x128
                                            ResBlock(self.bottom_dim, self.bottom_dim),
                                            nn.ReLU(),)
        self.decoder_bottom = nn.Sequential(nn.ConvTranspose2d(self.bottom_dim, self.bottom_dim, kernel_size=4, stride=2, padding=1),# 8x8x128
                                            ResBlock(self.bottom_dim, self.bottom_dim),
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(self.bottom_dim, self.top_dim, kernel_size=4, stride=2, padding=1),# 16x16x64
                                            ResBlock(self.top_dim, self.top_dim),
                                            nn.ReLU(),)
        self.decoder_top = nn.Sequential(nn.ConvTranspose2d(self.top_dim+self.bottom_dim, self.top_dim, kernel_size=4, stride=2, padding=1),# 32x32x64
                                            ResBlock(self.top_dim, self.top_dim),
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(self.top_dim, self.top_dim, kernel_size=4, stride=2, padding=1),# 64x64x64
                                            ResBlock(self.top_dim, self.top_dim),
                                            nn.Conv2d(self.top_dim, self.out_dim, kernel_size=1, stride=1),# 64x64x384
                                            nn.ReLU(),)
        
        self.pre_quantization_conv = nn.Conv2d(self.bottom_dim, self.embed_dim, kernel_size=1, stride=1)
        self.post_quantization_conv = nn.Conv2d(self.embed_dim, self.bottom_dim, kernel_size=1, stride=1)
        self.pre_quantization_conv2 = nn.Conv2d(self.top_dim*2, self.embed_dim, kernel_size=1, stride=1)
        self.post_quantization_conv2 = nn.Conv2d(self.embed_dim, self.top_dim, kernel_size=1, stride=1)
        self.quantizer_top = VectorQuantizer(n_e=2048, e_dim=64, beta=0.25)
        self.quantizer_bottom = VectorQuantizer(n_e=256, e_dim=64, beta=0.25)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear') # 4x4x128 -> 16x16x128 # nearest

    def forward(self, x):
        x = self.head(x) # 256x256x3 -> 64x64x32
        x_top = self.encoder_top(x) # 64x64x32 -> 16x16x64
        x_bottom = self.encoder_bottom(x_top) # 16x16x64 -> 4x4x128
        
        x_bottom = self.pre_quantization_conv(x_bottom) # 4x4x128 -> 4x4x64
        x_bottom_q, vq_loss_bottom, _, _, _ = self.quantizer_bottom(x_bottom)
        x_bottom_q = self.post_quantization_conv(x_bottom_q) # 4x4x64 -> 4x4x128
        
        x_bottom_dec = self.decoder_bottom(x_bottom_q) # 4x4x128 -> 16x16x64
        
        x_top = self.pre_quantization_conv2(torch.cat([x_bottom_dec, x_top], dim=1))  # 16x16x(128+64) -> 16x16x64  
        x_top_q, vq_loss_top, _, _, _ = self.quantizer_top(x_top)
        x_top_q = self.post_quantization_conv2(x_top_q) # 16x16x64 -> 16x16x64
        
        x_bottom_q = self.up(x_bottom_q) # 4x4x128 -> 16x16x128
        
        x_out = self.decoder_top(torch.cat([x_bottom_q, x_top_q], dim=1)) # 16x16x(128+64) -> 64x64x384
        return x_out, vq_loss_bottom+vq_loss_top
    

        
        