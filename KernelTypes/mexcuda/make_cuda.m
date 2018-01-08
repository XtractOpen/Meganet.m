
%%%%% WINDOWS (ERAN'S) COMPILATION:
%mexcuda -v '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64' -lcudnn mexGPUExample.cu
% mexcuda -v '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64' -lcudnn convCouple_mex.cu
%mexcuda -v -lcudnn convCouple_mex.cu % this actually works now (without the path)...

mexcuda -v -lcudnn convCuDNN2DSessionCreate_mex.cu convCuDNN2D.cu
mexcuda -v -lcudnn convCuDNN2DSessionDestroy_mex.cu convCuDNN2D.cu
mexcuda -v -lcudnn convCuDNN2D_mex.cu convCuDNN2D.cu
%%%%% Lars's COMPILATION:
% mexcuda -v  '-L /Users/lruthotto/Downloads/cuda/lib' -lcudnn convCouple_mex.cu
% mexcuda -v  -lcudnn mexGPUExample.cu