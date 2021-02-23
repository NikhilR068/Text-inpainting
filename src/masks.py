import torch
import torchvision as tv
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


def pixel_noise_mask(img, p=0.5):
    assert p > 0 and p < 1
    mask = torch.rand(512, 512)
    img[:, :, mask<p] = 0
    mask = mask > p
    mask = mask.repeat(1,3,1,1)
    return img, mask

def pixel_text_mask(for_image, sz=20, position=(128, 128), text='hello world'):

    """
    refer from:
    https://github.com/DmitryUlyanov/deep-image-prior/blob/master/utils/inpainting_utils.py

    """
    # number of texts and positions should be equal
    assert len(text) == len(position)
    font_fname = 'FreeSansBold.ttf'
    font_size = sz
    font = ImageFont.truetype(font_fname, font_size)
    img_mask = Image.fromarray(np.array(for_image)*0+255)
    
    binary_mask_temp = np.array(img_mask)
    binary_masks = np.zeros_like(binary_mask_temp, dtype=np.float32)
    binary_masks += 1.0
    binary_masks[binary_mask_temp<254] -= 1.0
    for i in range(len(text)):
        img_mask = Image.fromarray(np.array(for_image)*0+255)
        draw = ImageDraw.Draw(img_mask)
        # draw text 
        draw.text(position[i], text[i], font=font, fill='rgb(0, 0, 0)')
        binary_mask_temp = np.array(img_mask)
        binary_mask = np.zeros_like(binary_mask_temp, dtype=np.float32)
        binary_mask += 1.0
        binary_mask[binary_mask_temp<254] -= 1.0
        binary_masks = np.multiply(binary_masks, binary_mask)
    
    binary_masks = binary_masks.astype(np.uint8)
    binary_masks = Image.fromarray(binary_masks)
    # add mask to input image
    corrupted_image = np.multiply(for_image, binary_masks)
    corrupted_image = corrupted_image.astype(np.uint8)
    corrupted_image = Image.fromarray(corrupted_image)
    
    return corrupted_image,binary_masks
