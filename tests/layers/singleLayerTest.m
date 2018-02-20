classdef singleLayerTest < layerTest
    % classdef singleLayerTest < layerTest
    %
    % tests some single layers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            tvN   = getTVNormLayer([4 8 14]);
            ks{end+1} = singleLayer(dense([4*8 4]),'Bin',randn(4*8,3),'nLayer',tvN,'isWeight',0);
            ks{end+1} = singleLayer(dense([4*8 4]),'Bin',randn(4*8,3),'nLayer',tvN,'isWeight',0,'storeInterm',1);
            tvNt  = getTVNormLayer([4 8 14],'isWeight',1);
            ks{end+1} = singleLayer(dense([4*8 4]),'nLayer',tvNt);
            ks{end+1} = singleLayer(dense([4*8 4]),'nLayer',tvNt,'storeInterm',1);
            
            ks{end+1} = singleLayer(dense([24 14]),'Bin',rand(24,3));
            ks{end+1} = singleLayer(dense([24 14]),'Bin',rand(24,3),'storeInterm',1);
            ks{end+1} = singleLayer(dense([24 14],'Bin',eye(24)));
            ks{end+1} = singleLayer(dense([24 14],'Bin',eye(24)),'storeInterm',1);
            ks{end+1} = singleLayer(dense([24 14]),'storeInterm',1);
            ks{end+1} = singleLayer(dense([24 14]));
            ks{end+1} = singleLayer(convFFT([12 8], [3 3 2 5]));
            ks{end+1} = singleLayer(convFFT([12 8], [3 3 1 1]),'activation',@reluActivation);
            ks{end+1} = singleLayer(convFFT([12 8], [3 3 1 1]),'activation',@reluActivation,'storeInterm',1);
            ks{end+1} = singleLayer(dense([24 14],'useGPU',0,'precision','single'));
            ks{end+1} = singleLayer(dense([24 14],'useGPU',0,'precision','single'),'Bout',eye(24));
            ks{end+1} = singleLayer(dense([24 14],'useGPU',0,'precision','single'),'Bout',eye(24),'storeInterm',1);
            ks{end+1} = singleLayer(dense([24 14],'useGPU',0,'precision','single'));
            ks{end+1} = singleLayer(convFFT([12 8], [3 3 2 5],'useGPU',0,'precision','single'));
            ks{end+1} = singleLayer(convFFT([12 8], [3 3 2 5],'useGPU',0,'precision','single'),'storeInterm',1);
            ks{end+1} = singleLayer(convFFT([12 8], [3 3 1 1],'useGPU',0,'precision','single'),'activation',@reluActivation);
            testCase.layers = ks;
        end
    end
end