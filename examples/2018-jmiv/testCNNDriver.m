% =========================================================================
%
% Test file for cnnDriver. Sets up small-scale instances of the
% classification problem for stl10, cifar10, cifar100 with three different
% dynamics (parabolic, leapfrog, hamiltonian) and runs a few epochs.
%
% Use this test to make sure everything is set up. By default, we test on
% the CPU, but change the useGPU flag to test your GPU setup. 
%
% What to look for? If all is running correctly, you will see a slight (!)
% reduction of the objective function in all cases. The actual reduction
% varies from example to example and also depends on random initialization,
% but should exceed 10%.
%
% =========================================================================

datasets = {'stl10','cifar10','cifar100'};
dynamics = {'parabolic','leapfrog','hamiltonian'};

useGPU = 0;
precision = 'single';

augment = @(Y) randomCrop(randomFlip(Y,.5),4);
alpha = [2e-4; 2e-4; 2e-4];

for d1=1:numel(datasets)
    dataset = datasets{d1};
    for d2=1:numel(dynamics)
        dynamic = dynamics{d2};
        opt = sgd('learningRate',1e-2,'maxEpochs',10,'out',1,'miniBatch',10);
        [xc,His,xOptAcc]=cnnDriver(dataset,dynamic,100,100,2*[4 8 14 14],3,.1,useGPU,precision,opt,augment,alpha,[],[]);
    end
end