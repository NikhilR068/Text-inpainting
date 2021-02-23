import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from src.model import DIP_model
from src.masks import pixel_text_mask

device = 'cuda'

lr = 1e-2 #learning rate
inpaint_model = DIP_model() 
inpaint_model.to(device)
loss_func = nn.MSELoss() # loss function
optimizer = optim.Adam(inpaint_model.parameters(), lr=lr)

images = []
losses = []
to_tensor = tv.transforms.ToTensor()


x = Image.open('barbara.jpg')

# masked image and mask
corrupt_img, text_mask =  pixel_text_mask(np.array(x), sz=30, position=[(128, 128),(250,300)], 
                                        text=['Noct068','Inpainting'])
corrupt_img= to_tensor(corrupt_img).unsqueeze(0)
text_mask = to_tensor(text_mask).unsqueeze(0).bool()

corrupt_img = corrupt_img.to(device)
text_mask = text_mask.to(device)
#output placeholder
output_img = torch.Tensor(np.mgrid[:512, :512]).unsqueeze(0) / 512
output_img=output_img.to(device)

demo_corrupt = np.array(corrupt_img[0].cpu().detach().permute(1,2,0)*255, np.uint8)

print("starting Training")
for i in range(4000):
    optimizer.zero_grad()
    y = inpaint_model(output_img)
    loss = mse(corrupt_img, y*text_mask)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    if (i+1)%25 == 0 or i==0:
        with torch.no_grad():
            out = corrupt_img + y * ~text_mask
            out = out[0].cpu().detach().permute(1,2,0)*255
            out = np.array(out, np.uint8)
            images.append(out)
    
    if (i+1)%50==0:
        print('Iteration: {} Loss: {:.07f}'.format(i+1, losses[-1]))
print("Training finished...")

# Save final image

plt.imsave('denoised_image.jpg', out)

# GIF 
imageio.mimsave('denoise.gif', images)