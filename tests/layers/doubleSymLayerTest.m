classdef doubleSymLayerTest < layerTest
    % classdef singleLayerTest < layerTest
    %
    % tests some single layers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            ks{end+1} = doubleSymLayer(dense([14 14]),'Bin',randn(14,3));
            ks{end+1} = doubleSymLayer(dense([14 14]),'Bin',randn(14,3),'storeInterm',1);
            tvN   = getTVNormLayer([4 8 14]);
            ks{end+1} = doubleSymLayer(dense([4*8 4]),'Bin',randn(4*8,3),'nLayer',tvN);
            ks{end+1} = doubleSymLayer(dense([4*8 4]),'Bin',randn(4*8,3),'nLayer',tvN,'storeInterm',1);
            tvNt  = getTVNormLayer([4 8 14],'isWeight',1);
            ks{end+1} = doubleSymLayer(dense([4*8 4]),'nLayer',tvNt);
            ks{end+1} = doubleSymLayer(dense([4*8 4]),'nLayer',tvNt,'storeInterm',1);
            
            %             ks{2} = doubleSymLayer(dense([24 14]));
            ks{end+1} = doubleSymLayer(dense([14 14]),'Bout',randn(14,2),'useGPU',0,'precision','single');
            ks{end+1} = doubleSymLayer(dense([14 14]),'Bout',randn(14,2),'useGPU',0,'precision','single','storeInterm',1);
            ks{end+1} = doubleSymLayer(dense([24 14]),'useGPU',0,'precision','single','B2',randn(14,2));
            ks{end+1} = doubleSymLayer(dense([24 14]),'useGPU',0,'precision','single','B2',randn(14,2),'storeInterm',1);
            ks{end+1} = doubleSymLayer(dense([4,4]));
            ks{end+1} = doubleSymLayer(dense([4,4]),'storeInterm',1);
            testCase.layers = ks;
        end
    end
end