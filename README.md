
## Denoising 

Filling removed pixels with appropriate pixel for denoising
<div align='center'>
    <img src='images/lenna_ori.jpg' height="256">
    <img src='images/noisy-image.jpg' height="256">
    <img src='images/denoised-output.jpg' height="256">
</div>
Increasing training iteration may give a bit better output.

## Text Inpainting

Removing text masks from image
<div align='center'>
    <img src='images/barbara.jpg' height="256">
    <img src='images/text-corrupted-image.jpg' height="256">
    <img src='images/inpainting-output.jpg' height="256">
</div>


## Files

```
├── src                   # Compiled files (alternatively `dist`)
|    ├── masks.py
|    |   ├── FreeSansBold.ttf                   # font
|    |   ├── pixel noise mask                   # removing pixels from image
|    |   └── pixel text mask                    # add text mask to image
|    |
|    └── model.py                               # Pytorch Deep Image Prior Model 
|    
├── images                                      # input, masked and output images  
|
├── denoise_run.py                              # Train and save output denoised image
├── inpaint_run.py                              # Train and save output text inpainted image
├── Denoising and Inpainting.ipynb              # Jupyter Notebook with model output
├── inpaint.ipynb                               # Jupyter Notebook with tensorflow Pixel CNN for text inpainting
└── README.md

```

### RUN

```
python3 denoise_run.py
python3 inpaint_run.py
```
## Paper

**Dmitry Ulyanov et. al** *Deep Image Prior* [[arxiv](https://arxiv.org/abs/1711.10925)]