# QSM reconstruction w/ Deep Image Prior
To run code...
```
bash run_DIP.sh
```
I can not upload QSM dataset or demo QSM data due to privacy. 

## What is QSM?
Susceptibility
  - Response to an applied magnetic field
  - A material responses ⬆ or ⬇ in magnetic flux density

QSM (Quantitative Susceptibility Mapping)
  - Voxel intensity in QSM is value linearly proportional to underlying tissue apparent magnetic susceptibility.
  - Useful for chemical identification or quantification of specific biomarkers in contrast to structural MRI.

## How to obtain QSM?
<img src="https://github.com/9B8DY6/QSM_DIP__/assets/67573223/7ffec37d-9ee4-4832-8a45-b1e5ca955641" width=400, height=200>

For more details, please watch this [video](https://www.youtube.com/watch?v=TYWrFu464o8)

## Why is QSM reconstruction difficult?
In MRI, the local field  $`\delta B`$ induced by non-ferromagnetic biomaterial susceptibility along the main polarization $`B_0`$ field is the convolution of the volume susceptibility distribution $`\chi`$ with the dipole kernel $`d: \delta B = d \otimes \chi`$. This spatial convolution can be expressed as a point-wise multiplication in Fourier domain. $`\bigtriangleup B=D \cdot X`$. This Fourier expression provides an efficient way to predict the field perturbation when the the susceptibility distribution is known. However, the field to source inverse problem involves division by zero at a pair of cone surfaces at the magic angle with respect to $`B_0`$ in the Fourier domain. Consequently, susceptibility is <ins>**underdetermined**</ins> at the spatial frequencies on the cone surface, which often leads to severe streaking artifacts in the reconstructed QSM.

<img src="https://github.com/9B8DY6/QSM_DIP__/assets/67573223/c0663f28-a095-48fb-9b01-b33a9351b269" width=100, height=100>

From [QSM wikipedia](https://en.wikipedia.org/wiki/Quantitative_susceptibility_mapping)
## Method
In practice, there is no ground-truth in MRI acquisition but only gold-standard which is manually considered as standard. However, previous works solve this problem in supervised learning-way with regarding gold-standard as ground-truth. It is theoretically wrong approach so, we adopt Deep Image Prior which is optimization-based solver effective in inverse problems. 

Inverse Problem definition
```math
y=A(x)+b
```
$`y`$ is acquired data. $`x`$ is unknown true measurement. $`b`$ is a noise from sampling process. 

$`A: x \rightarrow y`$ is mapping operator from the true measurement to data. $`A`$ differs according to problem definition (denoising, deblurring, Super-Resolution, Inpainting, etc.)

Deep learning-based models learn inverse of $`A`$, $`A^{-1}`$ to reconstruct real signal $`x`$.
### Deep Image Prior (DIP)
[original paper](https://arxiv.org/abs/1711.10925)

A great deal of image statistics are captured by the structure of generator ConvNets, independent of learning. This is especially true for the statistics required to solve certain restoration problems, where image prior must supplement the information lost in the degradation processes. 

DIP interprets the neural network $`f_\theta`$ as a parameterziation of the image $`x`$. The network maps the paramters $`\theta`$, comprising the weights and bias of the filters in the network, to the image $`x`$. 

<img src="https://github.com/9B8DY6/QSM_DIP__/assets/67573223/a4623ead-c5a5-435b-a99e-5e2b143d3b00" width=400, height=200>

When you only have degraded image $`x_0`$, the network takes fixed gaussian noise vector $`z`$ as input and update it to reconstruct intact signal $`x`$ in $`x_0`$.
Then, the network $`f_\theta (z)`$ is optimized in way of parameterization of signal $`x`$ with respect to $`\theta`$.
The below formula satifies the optimization. 
```math
argmin_\theta ||A(f_\theta (z)) - y||^2_2
```

### My approach

<img src="https://github.com/9B8DY6/QSM_DIP__/assets/67573223/45c2e6a4-ee40-4ae3-9146-003f2a8ce349" width = 800, height = 200>

We only have degraded data, local-field $`\delta B`$. So we take $`N-`$ stacked 2D gaussian noise $`Z \in R^{N \times H \times W}`$ as input to 3D DIP to output $`x = f_\theta (z)`$. We optimize this 3D DIP with the formula  defined as below,
```math
argmin_\theta||d * f_\theta (z) - \delta B||
```
$`d`$ is dipole kernel. 
$`*`$ is spatial convolution. 

## Result
![image](https://github.com/9B8DY6/QSM_DIP__/assets/67573223/d2db58b4-0797-43d2-b659-e347437be3fe)

The upper one is "gold-standard" QSM and the below one is ours. We've got PSNR as 45.34. Ours have less distinct contrast than gold-standard but it is the first try to solve QSM reconstruction with Deep Image Prior.
