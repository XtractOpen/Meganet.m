classdef batchNormLayerTest < layerTest
	% classdef batchNormLayerTest < layerTest
	%
	% tests instance normalizationlayers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            ks{end+1} = batchNormLayer([12,14,5]);
            ks{end+1} = batchNormLayer([12,14,17],'isWeight',1);
            ks{end+1} = batchNormLayer([13,13,5],'precision','single');
            testCase.layers = ks;
        end
    end
end