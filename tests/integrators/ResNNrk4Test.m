classdef ResNNrk4Test < IntegratorTest
    % classdef ResNNTest < IntegratorTest
    %
    % tests some Residual Nets. Extend to cover more cases.
    methods (TestClassSetup)
        function addBlocks(testCase)
            ks    = cell(1,1);
            hY = rand(20,1); hY = hY/sum(hY);
            tY = [0;cumsum(hY)];
            % hth = rand(10,1); hth = hth/sum(hth);
            % tth = [0;cumsum(hth)];
            ks = cell(0,1);
%             ks{end+1} = ResNNrk4(singleLayer(dense([14 14]),‘activation’,@reluActivation),tth);
%             ks{end+1} = ResNNrk4(singleLayer(dense([14 14]),‘activation’,@reluActivation),tth,tY);
             TT = dense([14 14]);
             ks{end+1} = ResNNrk4(doubleLayer(TT,TT),tY); 
            
%             ks{3} = ResNNab3(singleLayer(affineTrafo(dense([14 14]))),3,0.1);
%             ks{4} = ResNNab3(singleLayer(affineTrafo(dense([14 14])),‘useGPU’,1),3,0.1);
%             ks{5} = ResNNab3(doubleLayer(TT,TT),4,.1,‘useGPU’,1,‘precision’,‘single’) ;
           testCase.integrators = ks;
        end
    end
end