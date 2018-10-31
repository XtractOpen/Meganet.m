%% all fail

runtests('instNormLayerTest') % ... instNormLayer doesn't exist
runtests('linearNegLayerTest') % ...linearNegLayer doesn't exist
runtests('convCuDNN2DTest') % - no mexcuda running
%%
EParabolic_STL10(5000,32,3) % --- last block creation issues

%% partial pass
runtests('MegaNetTest'); %  (1F)...adjoint is not matching...math error, maybe because I took out Wdata

%% all pass
runtests('NNTest') % just warnings from precision mismatches

runtests('singleLayerTest')
runtests('normLayerTest')
runtests('batchNormLayerTest')
runtests('tvNormLayerTest') 
runtests('doubleSymLayerTest')
runtests('doubleLayerTest')
runtests('affineScalingLayerTest')
runtests('ResNNTest');
runtests('LeapFrogNNTest');

runtests('convMCNTest');  % .... problem remains with padding
runtests('connectorTest');
runtests('DoubleHamiltonianNNTest');
runtests('IntegratorTest');
runtests('ConvFFTTest');
runtests('denseTest');
runtests('kernelTest');
runtests('scalingKernelTest');
runtests('sparseKernelTest');
runtests('convFFTTest') % MinExample
tb = runtests('layerTest');

%%
ECNN_MNIST_tf
ECNN_CIFAR10_tf
EResNN_Circle
E_ResNN_MNIST