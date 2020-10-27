% =========================================================================
%
% Driver for CIFAR-10 experiments described in
%
%
% Ruthotto L, Haber E: Deep Neural Networks motivated by PDEs,
%           Journal of Mathematical Imaging and Vision, 10.1007/s10851-019-00903-1, 2018
%
% 1) train with 40,000 training and 10,000 validation images (see Figs. 3 and 5)
% 2) train with 50,000 training images and no validation (see Table 1)
%
% Warning: This implementation is not geared to efficiency. Even with a resonable 
% GPU the computations will take several hours. 
% 
% =========================================================================

opt = sgd('nesterov',false,'ADAM',false,'miniBatch',32,'out',1,'lossTol',0.01);

lr = 0.1*1.5.^(-1:-1:-14)';
lr = [0.1*ones(40,1);  kron(lr,ones(10,1))];
lr = lr/2;

opt.learningRate     = @(epoch) lr(epoch);
opt.maxEpochs    = numel(lr);
opt.P        = @(x) min(max(x,-1),1);
opt.momentum = 0.9;

useGPU = 1;
precision = 'single';
alpha = [2e-4; 2e-4; 2e-4];

augmentCIFAR = @(Y) randomCrop(randomFlip(Y,.5),4);

%% train with validation set
cnnDriver('cifar10','parabolic',40000,10000,[32 64 112 112],3,1,useGPU,precision,opt,augmentCIFAR,alpha);
cnnDriver('cifar10','leapfrog',40000,10000,[32 64 112 112],3,1,useGPU,precision,opt,augmentCIFAR,alpha);
cnnDriver('cifar10','hamiltonian',40000,10000,[32 64 112 112],3,1,useGPU,precision,opt,augmentCIFAR,alpha);

%% train on all images
cnnDriver('cifar10','parabolic',50000,0,[32 64 112 112],3,1,useGPU,precision,opt,augmentCIFAR,alpha);
cnnDriver('cifar10','leapfrog',50000,0,[32 64 112 112],3,1,useGPU,precision,opt,augmentCIFAR,alpha);
cnnDriver('cifar10','hamiltonian',50000,0,[32 64 112 112],3,1,useGPU,precision,opt,augmentCIFAR,alpha);


