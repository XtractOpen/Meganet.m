classdef connectorTest < IntegratorTest
    % classdef connectorTest < IntegratorTest
    %
    % tests some connectors. Extend to cover more cases.
    
    methods (TestClassSetup)
        function addIntegrators(testCase)
            ks    = cell(1,1);
            ks{1} = connector(randn(24,10),randn(24,1));
%             ks{2} = ResNN(singleLayer(denseAntiSym([9 9])),4,.9);
%             TT = dense([14 14]);
%             ks{2} = ResNN(doubleLayer(TT,TT),4,.1) ;
            testCase.integrators = ks;
        end
    end
end