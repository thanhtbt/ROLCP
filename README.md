# ROLCP: A  fast randomized adaptive CP decomposition for streaming tensors
In this paper, we introduce a fast adaptive algorithm for CANDECOMP/PARAFAC decomposition of streaming three-way
tensors using randomized sketching techniques. By leveraging randomized least-squares regression and approximating
matrix multiplication, we propose an efficient first-order estimator to minimize an exponentially weighted recursive leastsquares cost function. Our algorithm is fast, requiring a low computational complexity and memory storage.

## Requirement 
Our MATLAB code requires the Tensor Toolbox http://www.tensortoolbox.org/

## DEMO 
Quick Start: Run the file DEMO.m

## State-of-the-art algorithms for comparison

+ PARAFAC_SDT, PARAFAC_RLST (2009): D. Nion et al. “Adaptive algorithms to track the PARAFAC decomposition of a third-order tensor,” IEEE Trans. Signal Process.,  2009.
+ SOAP (2017): N.V. Dung et al. “Second-order optimization based adaptive PARAFAC decomposition of three-way tensors,” Digit. Signal Process., 2017. 
+ OLCP (2016): S. Zhou et al. “Accelerating online CP decompositions for higher order tensors,”  ACM Int. Conf. Knowl. Discover. Data Min., 2016
+ OLSTEC (2017): H. Kasai, “Fast online low-rank tensor subspace tracking by CP decomposition using recursive least squares from incomplete observations,” Neurocomput., 2017

## Reference

This code is free and open source for research purposes. If you use this code, please acknowledge the following paper.

[1] L.T. Thanh, K. Abed-Meraim, N.L. Trung, A. Hafiance. "A fast randomized adaptive CP decomposition for streaming tensors". Int. Conf. Acoust. Speech  Signal Process. (ICASSP), 2021 (to appear).
