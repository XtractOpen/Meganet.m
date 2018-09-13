classdef tvNormLayerTest < layerTest
	% classdef tvNormLayerTest < layerTest
	%
	% tests instance normalizationlayers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            ks{end+1} = getTVNormLayer([3 4 7 12]);
            ks{end+1} = getTVNormLayer([3 4 5 22],'isWeight',1);
            testCase.layers = ks;
        end
    end
end