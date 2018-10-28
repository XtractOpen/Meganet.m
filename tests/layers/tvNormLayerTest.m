classdef tvNormLayerTest < layerTest
	% classdef tvNormLayerTest < layerTest
	%
	% tests instance normalizationlayers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            ks{end+1} = tvNormLayer([3 4 7]);
            ks{end+1} = tvNormLayer([3 4 5],'isWeight',1);
            testCase.layers = ks;
        end
    end
end