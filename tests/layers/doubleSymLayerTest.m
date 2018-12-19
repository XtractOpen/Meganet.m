classdef doubleSymLayerTest < layerTest
    % classdef singleLayerTest < layerTest
    %
    % tests some single layers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            ks{end+1} = doubleSymLayer(dense([14 14]),'Bin',randn(14,3));
            ks{end+1} = doubleSymLayer(dense([14 14]),'Bin',randn(14,3),'storeInterm',1);

            nImg = [4 8];
            sK   = [3 3 4 4];
            tvN   = tvNormLayer([nImg sK(4)]);
            Bin  = opCNNBias([nImg sK(end)]);
            Bout = opCNNBias([nImg sK(end)]);
            ks{end+1} = doubleSymLayer(convFFT(nImg,sK));
            ks{end+1} = doubleSymLayer(convFFT(nImg,sK),'normLayer1',tvN,'Bin',Bin,'Bout',Bout,'isWeight',0);
             
             
%             ks{end+1} = doubleSymLayer(dense([4*8 4]),'Bin',randn(4*8,3),'normLayer1',tvN,'storeInterm',1);
%             ks{end+1} = doubleSymLayer(dense([4*8 4*8]),'Bin',randn(4*8,3),'normLayer1',tvN,'normLayer2',tvN,'storeInterm',1);
%             tvNt  = getTVNormLayer([4 8 14],'isWeight',1);
%             ks{end+1} = doubleSymLayer(dense([4*8 4]),'normLayer1',tvNt);
%             ks{end+1} = doubleSymLayer(dense([4*8 4]),'normLayer1',tvNt,'storeInterm',1);
%             tvN2  = getTVNormLayer([3 2 14],'isWeight',1);
%             ks{end+1} = doubleSymLayer(dense([4*8 3*2]),'normLayer1',tvNt,'normLayer2',tvN2,'storeInterm',1);
%             
%             %             ks{2} = doubleSymLayer(dense([24 14]));
%             ks{end+1} = doubleSymLayer(dense([14 14]),'Bout',randn(14,2),'useGPU',0,'precision','single');
%             ks{end+1} = doubleSymLayer(dense([14 14]),'Bout',randn(14,2),'useGPU',0,'precision','single','storeInterm',1);
%             ks{end+1} = doubleSymLayer(dense([24 14]),'useGPU',0,'precision','single','B2',randn(14,2));
%             ks{end+1} = doubleSymLayer(dense([24 14]),'useGPU',0,'precision','single','B2',randn(14,2),'storeInterm',1);
%             ks{end+1} = doubleSymLayer(dense([4,4]));
%             ks{end+1} = doubleSymLayer(dense([4,4]),'storeInterm',1);
            testCase.layers = ks;
        end
    end
end