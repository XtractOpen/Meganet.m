classdef doubleSymLayerTest < layerTest
	% classdef singleLayerTest < layerTest
	%
	% tests some single layers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
               ks{end+1} = doubleSymLayer(dense([14 14]),'Bin',randn(14,3));
           tvN   = getTVNormLayer([4 8 14]);
                ks{end+1} = doubleSymLayer(dense([4*8 4]),'Bin',randn(4*8,3),'nLayer',tvN);
             tvNt  = getTVNormLayer([4 8 14],'isWeight',1);
               ks{end+1} = doubleSymLayer(dense([4*8 4]),'nLayer',tvNt);

            %             ks{2} = doubleSymLayer(dense([24 14]));
             ks{end+1} = doubleSymLayer(dense([14 14]),'Bout',randn(14,2),'useGPU',0,'precision','single');
             ks{end+1} = doubleSymLayer(dense([24 14]),'useGPU',0,'precision','single','B2',randn(14,2));
             ks{end+1} = doubleSymLayer(dense([4,4]));
            testCase.layers = ks;
        end
    end
end