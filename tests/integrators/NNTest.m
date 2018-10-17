classdef NNTest < IntegratorTest
    % classdef NNTest < IntegratorTest
    %
    % tests some neural networks. Extend to cover more cases.
    methods (TestClassSetup)
        function addIntegrators(testCase)
            ks    = cell(1,1);
            ks{1} = NN({singleLayer(dense([24 14]))});
            ks{2} = NN({singleLayer(dense([24 14])),singleLayer(dense([9 24])) }','outTimes',[1;1]);
            ks{3} = NN({singleLayer(convFFT([12 8],[3 3 2 5])),singleLayer(dense([10 5 12 8])) });
            ks{4} = NN({singleLayer(dense([24 14]))},'useGPU',0,'precision','single');
            ks{5} = NN({singleLayer(dense([24 14])),singleLayer(dense([9 24])) },'useGPU',0,'precision','single');
            ks{6} = NN({singleLayer(convFFT([12 8],[3 3 2 5])),singleLayer(dense([10 5 12 8])) },'useGPU',0,'precision','single');
            testCase.integrators = ks;
        end
    end
end