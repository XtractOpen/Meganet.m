classdef LeapFrogNNTest < IntegratorTest
    % classdef LeapFrogNNTest < blockTest
    %
    % tests some Residual Nets. Extend to cover more cases.
    
    methods (TestClassSetup)
        function addIntegrators(testCase)
            ks    = cell(0,1);
             ks{end+1} = LeapFrogNN(singleLayer(dense([14 14]),'activation',@reluActivation),3,0.1);
%             ks{2} = LeapFrogNN(singleLayer(denseAntiSym([9 9])),4,.9);
            TT = dense([4 4]);
              ks{end+1} = LeapFrogNN(doubleLayer(TT,TT),10,.1) ;
              ks{end+1} = LeapFrogNN(doubleSymLayer(TT),10,.1) ;
              ks{end+1} = LeapFrogNN(singleLayer(dense([14 14])),4,0.1);
           testCase.integrators = ks;
        end
    end
end