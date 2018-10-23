classdef ResNNTest < IntegratorTest
    % classdef ResNNTest < IntegratorTest
    %
    % tests some Residual Nets. Extend to cover more cases.
    
    methods (TestClassSetup)
        function addIntegrators(testCase)
            ks    = cell(0,1);
             ks{end+1} = ResNN(singleLayer(dense([14 14]),'activation',@reluActivation),3,0.1);
              ks{2} = ResNN(singleLayer(getDenseAntiSym([9 9])),4,.9);
            TT = dense([4 4]);
             ks{end+1} = ResNN(doubleLayer(TT,TT),10,.1) ;
             ks{end+1} = ResNN(doubleSymLayer(TT),10,.1) ;
             ks{end+1} = ResNN(singleLayer(dense([14 14])),4,0.1);
             ks{end+1} = ResNN(singleLayer(dense([14 14]),'useGPU',0),3,0.1);
             ks{end+1} = ResNN(doubleLayer(TT,TT),4,.1) ;
             ks{end+1} = ResNN(doubleSymLayer(TT),4,.1,'useGPU',0,'precision','double') ;
             
           testCase.integrators = ks;
        end
    end
end