# ROLCP: A  fast randomized adaptive CP decomposition for streaming tensors
In this work, we introduce a fast adaptive algorithm for CANDECOMP/PARAFAC decomposition of streaming three-way tensors using randomized sketching techniques. By leveraging randomized least-squares regression and approximating matrix multiplication, we propose an efficient first-order estimator to minimize an exponentially weighted recursive leastsquares cost function. Our algorithm is fast, requiring a low computational complexity and memory storage.

## Dependencies 
+ Our MATLAB code requires the [Tensor Toolbox](http://www.tensortoolbox.org/) which is already attached to this repository.
+ MATLAB 2019a

## Demo
Quick Start: Just run the file DEMO.m

## State-of-the-art algorithms for comparison

+ PARAFAC_SDT, PARAFAC_RLST (2009): D. Nion et al. “[Adaptive algorithms to track the PARAFAC decomposition of a third-order tensor](https://ieeexplore.ieee.org/document/4799120/),” IEEE Trans. Signal Process.,  2009.
+ SOAP (2017): N.V. Dung et al. “[Second-order optimization based adaptive PARAFAC decomposition of three-way tensors](https://www.sciencedirect.com/science/article/pii/S105120041730009X),” Digit. Signal Process., 2017. 
+ OLCP (2016): S. Zhou et al. “[Accelerating online CP decompositions for higher order tensors](https://dl.acm.org/doi/10.1145/2939672.2939763),”  ACM Int. Conf. Knowl. Discover. Data Min., 2016
+ OLSTEC (2017): H. Kasai, “[Fast online low-rank tensor subspace tracking by CP decomposition using recursive least squares from incomplete observations](https://www.sciencedirect.com/science/article/pii/S0925231218313584),” Neurocomput., 2017

## Some Results

Running time and estimation accuracy of adaptive CP algorithms

<p float="left">
  <img src="https://user-images.githubusercontent.com/26319211/110488183-87920e80-80ee-11eb-9c66-42d212d07382.jpg" width="350" height='250' />
  <img src="https://user-images.githubusercontent.com/26319211/110486987-7399dd00-80ed-11eb-8163-33b9edcef365.PNG" width="300" height='250' /> 
</p>


## Reference

This code is free and open source for research purposes. If you use this code, please acknowledge the following paper.

[1] L.T. Thanh, K. Abed-Meraim, N.L. Trung, A. Hafiance. "[*A fast randomized adaptive CP decomposition for streaming tensors*](https://drive.google.com/file/d/1DAUTPryASpIoDxUZlRW_jzMSFeOS5EPm/view)". **IEEE Int. Conf. Acoust. Speech  Signal Process. (IEEE ICASSP)**, 2021.

