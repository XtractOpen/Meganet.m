%% all fail

runtests('instNormLayerTest') % ... instNormLayer doesn't exist
runtests('linearNegLayerTest') % ...linearNegLayer doesn't exist
runtests('convCuDNN2DTest') % - no mexcuda running
%%
EParabolic_STL10(5000,32,3) 

% dnnVarProObjFctn
% softmaxLoss


%% partial pass

%% all pass
runtests('NNTest') % just warnings from precision mismatches

runtests('MegaNetTest');
runtests('singleLayerTest') % testGetJYOp tempermental
runtests('normLayerTest')
runtests('batchNormLayerTest')
runtests('tvNormLayerTest') 
runtests('doubleSymLayerTest')
runtests('doubleLayerTest')
runtests('affineScalingLayerTest') % CUDA issues
runtests('ResNNTest');
runtests('LeapFrogNNTest');
runtests('convMCNTest');
runtests('connectorTest');
runtests('DoubleHamiltonianNNTest');
runtests('IntegratorTest');
runtests('ConvFFTTest');
runtests('denseTest'); % testAdjoint tempermental
runtests('kernelTest');
runtests('scalingKernelTest');
runtests('sparseKernelTest');
runtests('convFFTTest') % MinExample
tb = runtests('layerTest');

%%
ECNN_MNIST_tf
EResNN_Peaks

ECNN_CIFAR10_tf
EResNN_Circle
E_ResNN_MNIST