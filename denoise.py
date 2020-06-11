import os

import imageio

import numpy as np
import torch

import model

import matplotlib.pyplot as plt

# select color or gray-scale
# color = 'gray'
color = 'color'

# define the noise level
sigma = 25

# load a test image
y = imageio.imread(os.path.join('data','water-castle.png')).astype(np.float32)/255
if color == 'gray':
    y = np.mean(y, 2, keepdims=True)
# add noise
z = y + sigma/255. * np.random.randn(*y.shape).astype(np.float32)

# load the model state dict
checkpoint = torch.load(os.path.join('checkpoints', f'tdv3-3-25-f32-{color}.pth'))
sigma_ref = 25

# get the variational network with the TDV regularizer
vn = model.VNet(checkpoint['config'], efficient=False)
vn.load_state_dict(checkpoint['model'])
vn.cuda()

# define the evaluation metric
def psnr(x, y): 
    return 20*np.log10(1.0/np.sqrt(np.mean((x-y) ** 2)))

# define the application of the VN
def apply_vn(x_0, z):
    # tranform to reference noise level
    scale = sigma_ref/sigma
    x = vn(x_0 * scale, z * scale)
    # convert back to original scale
    x = [j/scale for j in x]
    return x

# push the images to torch
y_th = torch.from_numpy(np.transpose(y, (2,0,1))[None]).cuda()
z_th = torch.from_numpy(np.transpose(z, (2,0,1))[None]).cuda()

with torch.no_grad():
    x_th = apply_vn(z_th, z_th)

x_S = np.transpose(x_th[-1][0].cpu().numpy(), (1,2,0))

# show the result
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
ax[0].imshow(z.squeeze(), vmin=0, vmax=1, cmap='gray')
ax[0].set_title('z')
ax[0].set_xlabel(f'PSNR={psnr(z,y):.2f}dB')
ax[1].imshow(x_S.squeeze(), vmin=0, vmax=1, cmap='gray')
ax[1].set_title('x_S')
ax[1].set_xlabel(f'PSNR={psnr(x_S,y):.2f}dB')
ax[2].imshow(y.squeeze(), vmin=0, vmax=1, cmap='gray')
ax[2].set_title('y')

plt.show()
