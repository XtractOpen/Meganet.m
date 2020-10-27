% =========================================================================
%
% Driver for STL-10 experiments described in
%
%
% Ruthotto L, Haber E: Deep Neural Networks motivated by PDEs,
%           Journal of Mathematical Imaging and Vision, 10.1007/s10851-019-00903-1, 2018
%
% 1) experiment with reduced training sets (see Figure 4)
% 2) train with 4,000 training and 1,000 validation images (see Figs. 3 and 5)
% 3) train with 5,000 training images and no validation (see Table 1)
%
% Warning: This implementation is not geared to efficiency. Even with a resonable 
% GPU the computations will take several hours. 
% 
% =========================================================================

opt = sgd('nesterov',false,'ADAM',false,'miniBatch',32,'out',1,'lossTol',0.01);

lr = 0.1*1.5.^(-1:-1:-14)';
lr = [0.1*ones(40,1);  kron(lr,ones(10,1))]/2;


opt.learningRate   = @(epoch) lr(epoch);
opt.maxEpochs      = numel(lr);
opt.P              = @(x) min(max(x,-1),1);
opt.momentum       = 0.9;

useGPU = 1;
precision = 'single';

augmentSTL   = @(Y) randomCrop(randomFlip(Y,.5),12);
augmentOFF   = @(Y) Y;

nf = [16 32 64 128 128];


%% experiment with reduced data set
nex = 500:500:4000;
for k=1:numel(nex)
    cnnDriver('stl10','parabolic',nex(k),1000,nf,5,1,useGPU,precision,opt,augmentOFF);
	cnnDriver('stl10','leapfrog',nex(k),1000,nf,5,1,useGPU,precision,opt,augmentOFF);
	cnnDriver('stl10','hamiltonian',nex(k),1000,nf,5,1,useGPU,precision,opt,augmentOFF);
end


%% train with validation
cnnDriver('stl10','hamiltonian',4000,1000,nf,5,1,useGPU,precision,opt,augmentSTL);
cnnDriver('stl10','leapfrog',4000,1000,nf,5,1,useGPU,precision,opt,augmentSTL);
cnnDriver('stl10','parabolic',4000,1000,nf,5,1,useGPU,precision,opt,augmentSTL);

%% train on all examples
cnnDriver('stl10','hamiltonian',5000,0,nf,5,1,useGPU,precision,opt,augmentSTL);
cnnDriver('stl10','leapfrog',5000,0,nf,5,1,useGPU,precision,opt,augmentSTL);
cnnDriver('stl10','parabolic',5000,0,nf,5,1,useGPU,precision,opt,augmentSTL);


