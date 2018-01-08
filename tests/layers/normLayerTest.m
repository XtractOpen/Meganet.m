classdef normLayerTest < layerTest
	% classdef batchNormLayerTest < layerTest
	%
	% tests instance normalizationlayers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            ks{end+1} = normLayer([20,3,12],'doNorm',[1,0,0]);
            ks{end+1} = normLayer([20,3,12],'doNorm',[0,1,0]);
            ks{end+1} = normLayer([20,3,12],'doNorm',[0,0,1]);
            ks{end+1} = normLayer([20,3,12],'doNorm',[0,1,1]);
            ks{end+1} = normLayer([20,3,12],'doNorm',[1,1,1]);
            testCase.layers = ks;
        end
    end
    
%     methods (Test)
% 
%     end
end