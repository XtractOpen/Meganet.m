%% all fail


runtests('instNormLayerTest') % ... dims don't match, Array dimensions must match for binary array op.
runtests('linearNegLayerTest') % ... dims don't match

runtests('convCuDNN2DTest') % - no mexcuda running
EParabolic_STL10(5000,32,3)

%% partial pass
runtests('normLayerTest') % ... dims
%%
runtests('batchNormLayerTest') % .... dimension issues
%%
runtests('tvNormLayerTest') %  ... Array dimensions must match for binary array op.
%%
runtests('doubleLayerTest') % ...math error
%%
runtests('doubleSymLayerTest') % ... dims don't match

%% all pass
runtests('NNTest') % just warnings from precision mismatches
runtests('affineScalingLayerTest')
runtests('singleLayerTest')

runtests('MegaNetTest'); %  (4P,2F)...dense layers testVecInput issues
runtests('convMCNTest');  % .... problem remains with padding
runtests('connectorTest');
runtests('DoubleHamiltonianNNTest');
runtests('IntegratorTest');
runtests('LeapFrogNNTest');
runtests('ResNNTest');
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