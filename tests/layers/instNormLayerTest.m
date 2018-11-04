classdef instNormLayerTest < layerTest
	% classdef instNormLayerTest < layerTest
	%
	% tests instance normalizationlayers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            ks{end+1} = instNormLayer([2 5 12]);
            ks{end+1} = instNormLayer([2 5 12],'isWeight',1);
            testCase.layers = ks;
        end
    end
end