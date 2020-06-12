import os

import imageio
from skimage.filters import gaussian

import numpy as np
import torch

import model

import matplotlib.pyplot as plt

# select color or gray-scale
# color = 'gray'
color = 'color'

# define the number of gradient steps and step size
num_iter = 1000
tau = 0.05

# load a test image
x_orig = imageio.imread(os.path.join('data','face.jpg')).astype(np.float32)/255
if color == 'gray':
    x_orig = np.mean(x_orig, 2, keepdims=True)

# load the model state dict
checkpoint = torch.load(os.path.join('checkpoints', f'tdv3-3-25-f32-{color}.pth'))

# get the variational network with the TDV regularizer
vn = model.VNet(checkpoint['config'], efficient=False)
vn.load_state_dict(checkpoint['model'])
vn.cuda()
print(vn.R)
# fix the parameters
for p in vn.R.parameters():
    p.requires_grad_(False)

# push the images to torch
x = torch.from_numpy(np.transpose(x_orig, (2,0,1))[None]).cuda()

# create mask to smooth out the boundary
m, n, c = x_orig.shape
pad = 50
mask = np.zeros((m+2*pad,n+2*pad), dtype=np.float32)
mask[pad//2:-pad//2,pad//2:-pad//2] = 1
mask = gaussian(mask, sigma=pad//4)
mask = torch.from_numpy(mask[None,None]).cuda()

# define the objective
def loss(x):
    grad_R = vn.R.grad(x)
    # compute the Rayleigh quotient
    Lambda = torch.sum((x)* x, (1,2,3), keepdim=True) / torch.sum(x**2, (1,2,3), keepdim=True)
    return torch.mean((grad_R - Lambda * x)**2)

# plotting
plt.ion()
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].imshow(x_orig.squeeze(), cmap='gray')

x = torch.nn.functional.pad(x , (pad,pad,pad,pad), 'replicate') * mask
x_old = x.clone()
const_ = torch.mean(x**2, (1,2,3), True).sqrt_()
for j in range(num_iter+1):
    # overrelax
    x_bar = (x + (j-1)/(j+2) * (x - x_old)).detach_().requires_grad_(True)
    x_old = x.clone()
    # loss function
    l = loss(x_bar)
    if not x_bar.grad is None: x_bar.grad.zero_()
    l.backward()
    # projected gradient step on x
    x = x_bar.data - tau*x_bar.grad
    x = torch.clamp(x, 0, 1)

    if j % 25 == 0:
        print(f'{j:03d}: l={l.item():.3f}')
        x_np = np.transpose(x[0].cpu().numpy(), (1,2,0))[pad:-pad,pad:-pad,:]
        ax[1].imshow(x_np.squeeze(), cmap='gray')
        fig.canvas.draw()
        plt.pause(0.001)

plt.ioff()
plt.show()
