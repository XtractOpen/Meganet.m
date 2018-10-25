classdef DoubleHamiltonianNNTest < IntegratorTest
    % classdef ResNNTest < blockTest
    %
    % tests some Residual Nets. Extend to cover more cases.
    
    methods (TestClassSetup)
        function addIntegrators(testCase)
            ks    = cell(0,1);
            layer = doubleSymLayer(dense([2,2]));
             ks{end+1}   = DoubleHamiltonianNN(layer,layer,10,.1);
             layer1 = doubleSymLayer(dense([3,3]));
             layer2 = doubleSymLayer(dense([1,1]));
             ks{end+1}   = DoubleHamiltonianNN(layer,layer,100,.1);
            
           testCase.integrators = ks;
        end
    end
end