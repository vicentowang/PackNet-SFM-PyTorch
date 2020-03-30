from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        Ri = R.transpose(1, 2)
        ti = -1 * t
        Ti = get_translation_matrix(ti)

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(Ri, Ti)
        Mi = torch.matmul(T, R)
    else:
        M = torch.matmul(T, R)
        Mi = M

    return M, Mi


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.norm = nn.GroupNorm(16, out_channels, 1e-10)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        z = (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = cam_points[:, :2, :] / z

        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords, z.view(self.batch_size, 1, self.height, self.width)









def Space2Depth_O(x, r=2):
    """Takes a BCHW tensor and returns a B(rC)(H/r)(W/r) tensor,
    by concatenating neighbor spatial pixels as extra channels.
    It is the inverse of nn.PixelShuffle (if you apply both sequentially you should get the same tensor)
    Example r=2: A RGB image (C=3) becomes RRRRGGGGBBBB (C=12) and is downsampled to half its size
    """
    b, c, h, w = x.shape
    out_channel = c * (r ** 2)
    out_h, out_w = h // r, w // r
    x = x.contiguous().view(b, c, out_h, r, out_w, r)
    return x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)

class Space2Depth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C *(self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x
    
class Conv_3D(nn.Module):
    def __init__(self, out_channels = 8, kernel_size = 3, in_channels = 1):
        super().__init__()
        self.conv_3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding = (kernel_size - 1)//2)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)
        return self.conv_3d(x)

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, D, C, H, W = x.shape
        return x.view(N, D*C ,H , W)

class Conv_2D(nn.Module):
    def __init__(self, in_channels , out_channels , kernel_size = 3):
        super().__init__()
        self.conv_2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding = (kernel_size - 1)//2)

    def forward(self, x):
        return self.conv_2d(x)

class Packing(nn.Module):

    def __init__(self, in_channels, out_channels, LR_times = 2, kernel_size = 3, depth_3dconv = 8):
        super().__init__()
        self.space_to_depth = Space2Depth(LR_times)
        self.conv_3d = Conv_3D(depth_3dconv, kernel_size)
        self.reshape = Reshape()
        self.conv_2d = Conv_2D((LR_times**2)*in_channels*depth_3dconv, out_channels, kernel_size)

    def forward(self, x):
        x = self.space_to_depth(x)
        x = self.conv_3d(x)
        x = self.reshape(x)
        x = self.conv_2d(x)      
        return x

class Depth2Space(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.depth_to_space = nn.PixelShuffle(block_size)

    def forward(self, x):
        return self.depth_to_space(x);
    
class UnPacking(nn.Module):

    def __init__(self, in_channels, out_channels, HR_times = 2, kernel_size = 3, depth_3dconv = 8):
        super().__init__()
        self.conv_2d = Conv_2D(in_channels, (HR_times**2)*out_channels//depth_3dconv, kernel_size)
        self.conv_3d = Conv_3D(depth_3dconv, kernel_size)
        self.reshape = Reshape()
        self.depth_to_space = Depth2Space(HR_times)
        
    def forward(self, x):
        x = self.conv_2d(x)
        x = self.conv_3d(x)
        x = self.reshape(x)
        x = self.depth_to_space(x)
        return x
  
class Conv2d(nn.Module):
    def __init__(self, in_channels , out_channels , kernel_size = 3, GN = 16):
        super().__init__()
        self.conv_2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding = (kernel_size - 1)//2)
        self.group_norm = nn.GroupNorm(GN, out_channels)
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        x = self.conv_2d(x)
        x = self.group_norm(x)
        x = self.elu(x)
        return x
    
class InvDepth(nn.Module):
    def __init__(self, in_channels , out_channels , kernel_size = 3):
        super().__init__()
        self.conv_2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding = (kernel_size - 1)//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_2d(x)
        x = self.sigmoid(x)
        return x  
    
class ResidualBlock(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()       
        self.conv2d_1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        self.conv2d_2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.conv2d_3 = torch.nn.Conv2d(out_channels, out_channels, 1)
        self.group_norm = nn.GroupNorm(16, out_channels)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.5)
  
    def forward(self, x):
       
        residual = x
        
        out = self.conv2d_1(x)
        out = self.elu(out)

        out = self.conv2d_2(out)
        out = self.elu(out)
        
        out = self.conv2d_3(out)
        out = self.group_norm(out)
        
        out += residual     
        out = self.elu(out)
        out = self.dropout(out)
        return out

def Upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")   
    
class PackNetsEncoder(nn.Module):

    def __init__(self, num_input_images=1):
        super(PackNetsEncoder, self).__init__()
        self.layer1 = nn.Sequential( Conv2d(3, 64, kernel_size = 5) )
        self.layer2 = nn.Sequential( Conv2d(64, 64, kernel_size = 7), Packing(64, 64) )
        self.layer3 = nn.Sequential( ResidualBlock(64,64),ResidualBlock(64,64), Packing(64, 64) )
        self.layer4 = nn.Sequential( ResidualBlock(64,64),ResidualBlock(64,64), Packing(64, 128) )
        self.layer5 = nn.Sequential( ResidualBlock(128,128),ResidualBlock(128,128),ResidualBlock(128,128), Packing(128, 256) )
        self.layer6 = nn.Sequential( ResidualBlock(256,256),ResidualBlock(256,256),ResidualBlock(256,256), Packing(256, 512) )
        #self.encoder_nets = nn.Sequential(self.layer1,self.layer2,self.layer3,self.layer4,self.layer5,self.layer6)
        
        self.layer7 =   nn.ModuleDict({ 'Unpacking': UnPacking(512, 256) , 'Conv2d': Conv2d(512, 512) })
        self.layer8 =   nn.ModuleDict({ 'Unpacking': UnPacking(512, 256) , 'Conv2d': Conv2d(384, 256) })
        self.layer9 =   nn.ModuleDict({ 'InvDepth' : InvDepth(256, 1) })
        
        self.layer10 =  nn.ModuleDict({ 'Unpacking': UnPacking(1, 64) , 'Conv2d': Conv2d(129, 128) })
        self.layer11 =  nn.ModuleDict({ 'InvDepth' : InvDepth(128, 1) })
        
        self.layer12 =  nn.ModuleDict({ 'Unpacking': UnPacking(1, 64) , 'Conv2d': Conv2d(129, 64) })
        self.layer13 =  nn.ModuleDict({ 'InvDepth' : InvDepth(64, 1) })
        
        self.layer14 =  nn.ModuleDict({ 'Unpacking': UnPacking(1, 64) , 'Conv2d': Conv2d(129, 64) })
        self.layer15 =  nn.ModuleDict({ 'InvDepth' : InvDepth(64, 1) })
  
    def forward(self, x0):
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5) # x6[N ,512, H/32， W/32]
        
        x7 = self.layer7['Conv2d'](torch.cat( (x5,self.layer7['Unpacking'](x6)) , 1))
        x8 = self.layer8['Conv2d'](torch.cat( (x4,self.layer8['Unpacking'](x7)) , 1))
        x9 = self.layer9['InvDepth'](x8)
        
        x10 = self.layer10['Conv2d'](torch.cat( (x3,self.layer10['Unpacking'](x9),Upsample(x9)) , 1))
        x11 = self.layer11['InvDepth'](x10)
        
        x12 = self.layer12['Conv2d'](torch.cat( (x2,self.layer12['Unpacking'](x11),Upsample(x11)) , 1))       
        x13 = self.layer13['InvDepth'](x12)
        
        x14 = self.layer14['Conv2d'](torch.cat( (x1,self.layer12['Unpacking'](x13),Upsample(x13)) , 1))
        x15 = self.layer15['InvDepth'](x14)
        
        xout = x15
        return xout # x15[N , 1, H ,W]










  
class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C *(self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x
    
class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x
# class ReflectionPad3d(nn.Module):
#     ''' Implements 3d version of ReflectionPad2d'''
#     def __init__(self, padding):
#         super(ReflectionPad3d, self).__init__()
#         self.padding = torch.nn.modules.utils._ntuple(6)(padding)
#         self.pad3d = torch.nn.modules.padding._ReflectionPadNd.apply
#
#     def forward(self, input):
#         x = self.pad3d(input, self.padding)
#         return x

class ReflectionPad3d(nn.Module):

    def __init__(self, padding):
        super(ReflectionPad3d, self).__init__()
        self.padding = torch.nn.modules.utils._ntuple(6)(padding)

    def forward(self, x):
        return F.pad(x,self.padding, mode='reflect')


class PackBlock(nn.Module):
    def __init__(self, in_channels, out_channels,use_conv3d=True,use_refl=False, need_last_nolin=True):
        super(PackBlock, self).__init__()
        self.useconv3d = use_conv3d
        self.need_last_nolin = need_last_nolin

        self.downscaled_channel = int(in_channels // 2)
        self.conv2d_1 = nn.Conv2d(in_channels * 4, self.downscaled_channel, 1, bias=False)
        self.norm2d_1 = nn.GroupNorm(16, self.downscaled_channel, 1e-10)
        self.nolin2d_1 = nn.ELU(inplace=True)

        if use_conv3d:
            self.conv3d_channel = 4
            self.conv3d = nn.Conv3d(1, self.conv3d_channel, 3, bias=True)
            self.nolin3d = nn.ELU(inplace=True)

            self.conv2d = nn.Conv2d(np.int(self.downscaled_channel * self.conv3d_channel), out_channels, 3, bias=False)
        else:
            self.conv2d = nn.Conv2d(self.downscaled_channel, out_channels, 3, bias=False)

        if use_refl:
            self.pad2d = nn.ReflectionPad2d(1)
            self.pad3d = nn.ReplicationPad3d(1)
        else:
            self.pad2d = nn.ZeroPad2d(1)
            self.pad3d = nn.ConstantPad3d(1, 0)


        self.norm2d = nn.GroupNorm(16, out_channels, 1e-10)

        self.pool3d = SpaceToDepth(2)
        self.nolin = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.pool3d(x)

        # we do down channel first
        x = self.conv2d_1(x)
        x = self.norm2d_1(x)
        x = self.nolin2d_1(x)

        N, C, H, W = x.size()
        x = x.view(N, 1, C, H, W)
        if self.useconv3d:
            x = self.pad3d(x)
            x = self.conv3d(x) # now is [N, 8, C, H, W]
            x = self.nolin3d(x)
            C *= self.conv3d_channel

        x = x.view(N, C, H, W)
        x = self.pad2d(x)
        x = self.conv2d(x)
        x = self.norm2d(x)
        if self.need_last_nolin:
            x = self.nolin(x)
        return x

class UnPackBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv3d=True, use_refl=True, need_last_nolin=True):
        super(UnPackBlock, self).__init__()
        self.useconv3d = use_conv3d
        self.need_last_nolin = need_last_nolin

        self.nolin = nn.ELU(inplace=True)

        if use_conv3d:
            self.conv3d_channels = 4
            self.conv3d = nn.Conv3d(1, self.conv3d_channels, 3, bias=True)
            self.nolin3d = nn.ELU(inplace=True)
            self.conv2d = nn.Conv2d(in_channels, np.int(4 * out_channels / self.conv3d_channels), 3, bias=True)
            self.norm2d = nn.GroupNorm(16, np.int(4 * out_channels / self.conv3d_channels), 1e-10)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels * 4, 3, bias=True)
            self.norm2d = nn.GroupNorm(16, out_channels * 4, 1e-10)

        if use_refl:
            self.pad2d = nn.ReflectionPad2d(1)
            self.pad3d = nn.ReplicationPad3d(1)
        else:
            self.pad2d = nn.ZeroPad2d(1)
            self.pad3d = nn.ConstantPad3d(1, 0)

        self.upsample3d = DepthToSpace(2)

    def forward(self, x):
        x = self.pad2d(x)
        x = self.conv2d(x)
        x = self.norm2d(x)
        x = self.nolin(x)

        # B * 4Co/D * H * W

        N, C, H, W = x.size()

        if self.useconv3d:
            x = x.view(N, 1, C, H, W)
            x = self.pad3d(x)
            x = self.conv3d(x)
            x = self.nolin3d(x)
            x = x.view(N, C * self.conv3d_channels, H, W)

        x = self.upsample3d(x)
        return x

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    
def testing():
    x = torch.from_numpy(np.linspace(1,192,192))
    x = x.view(1,3,8,8)

    x = torch.randn(1, 3, 64, 64)
    x = x.float()
    pack = PackNetsEncoder()
    out = pack(x)
    net = Packing(3, 12)
    for parameter in net.parameters():
        print(parameter.shape)
        
    x1 = Packing(3, 12)(x)
    x2 = UnPacking(12, 4)(x1)
    return x2
if __name__ == "__main__":
    print(testing().shape)
