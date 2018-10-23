%% all fail
runtests('NNTest') % - dense layer on top of conv layer...dims don't match up. Recall, you say dense will need a reshape....where does that go?
runtests('convCuDNN2DTest') % - no mexcuda running
runtests('affineScalingLayerTest') %
runtests('doubleLayerTest') % ... dims don't match, dense on conv layer
runtests('doubleSymLayerTest') % ... dims don't match
runtests('instNormLayerTest') % ... dims don't match, Array dimensions must match for binary array op.
runtests('linearNegLayerTest') % ... dims don't match
runtests('singleLayerTest') % ... dims
runtests('convFFTTest') % MinExample
EParabolic_STL10

%% partial pass
runtests('normLayerTest') % ... dims
%%
runtests('batchNormLayerTest') % .... dimension issues
%%
runtests('tvNormLayerTest') %  ... Array dimensions must match for binary array op.

%% all pass
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
tb = runtests('layerTest');

%%
ECNN_MNIST_tf
ECNN_CIFAR10_tf
EResNN_Circle
E_ResNN_MNIST