import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from src.model import DIP_model
from src.masks import pixel_noise_mask

device = 'cuda'

lr = 1e-2 # learning rate
denoise_model = DIP_model()
denoise_model.to(device)
loss_func = nn.MSELoss() # loss function
optimizer = optim.Adam(denoise_model.parameters(), lr=lr)

images = []
losses = []
to_tensor = tv.transforms.ToTensor()
img = Image.open('lenna_ori.png')
input_img  = to_tensor(img).unsqueeze(0)

# masked image and mask
corrupt_img, noise_mask = pixel_noise_mask(input_img, 0.8)
noise_mask = noise_mask.to(device)
corrupt_img = corrupt_img.to(device)
#output placeholder
output_img = torch.Tensor(np.mgrid[:512, :512]).unsqueeze(0) / 512
output_img=output_img.to(device)

demo_corrupt = np.array(corrupt_img[0].cpu().detach().permute(1,2,0)*255, np.uint8)


# training 

print("Training Starting...")
for i in range(4000):
    optimizer.zero_grad()
    output_img = denoise_model(denoise_img)
    loss = loss_func(corrupt_img, output_img*noise_mask)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    if (i+1)%25 == 0 or i==0:
        with torch.no_grad():
            out = corrupt_img + output_img * ~noise_mask
            out = out[0].cpu().detach().permute(1,2,0)*255
            out = np.array(out, np.uint8)
            images.append(out)
    if (i+1)%50==0:
        print('Iteration: {} Loss: {:.07f}'.format(i+1, losses[-1]))
        
print("Fininshed Denoising...")

# Save final image

plt.imsave('denoised_image.jpg', out)

# GIF 
imageio.mimsave('denoise.gif', images)
