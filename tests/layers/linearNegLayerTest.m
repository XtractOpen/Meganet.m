classdef linearNegLayerTest < layerTest
	% classdef singleLayerTest < layerTest
	%
	% tests some single layers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(1,1);
            ks{1} = linearNegLayer(dense([24 14]));
            ks{2} = linearNegLayer(convFFT([12 8], [3 3 2 5]));
            testCase.layers = ks;
        end
    end
end