classdef DoubleHamiltonianNNTest < IntegratorTest
    % classdef ResNNTest < blockTest
    %
    % tests some Residual Nets. Extend to cover more cases.
    
    methods (TestClassSetup)
        function addIntegrators(testCase)
            ks    = cell(0,1);
            K     = convFFT([24 24],[3 3 4 8]);
            layer = doubleSymLayer(K);
             ks{end+1}   = DoubleHamiltonianNN(layer,layer,10,.1);
%              ks{end+1}   = DoubleHamiltonianNN(layer,layer,100,.1);
            
           testCase.integrators = ks;
        end
    end
end