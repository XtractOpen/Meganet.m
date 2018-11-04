classdef doubleLayerTest < layerTest
    % classdef doubleLayerTest < layerTest
    %
    % tests some double layers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            nImg = [4 8];
            sK   = [3 3 4 4];
            tvN   = tvNormLayer([nImg sK(4)]);
            ks{end+1} = doubleLayer(convFFT(nImg,sK),convFFT(nImg,sK));
            ks{end+1} = doubleLayer(convFFT(nImg,sK),convFFT(nImg,sK),'normLayer1',tvN,'normLayer2',tvN);
            ks{end+1} = doubleLayer(convFFT(nImg,sK),convFFT(nImg,sK),'normLayer1',tvN,'normLayer2',tvN,'storeInterm',1);
            
            Bin1 = opCNNBias([nImg sK(end)]);
            Bin2 = opCNNBias([nImg sK(end)]);
            Bout = opCNNBias([nImg sK(end)]);
            ks{end+1} = doubleLayer(convFFT(nImg,sK),convFFT(nImg,sK),'Bin1',Bin1);
            ks{end+1} = doubleLayer(convFFT(nImg,sK),convFFT(nImg,sK),'Bin2',Bin2,'Bout',Bout);
            ks{end+1} = doubleLayer(convFFT(nImg,sK),convFFT(nImg,sK),'Bin1',Bin1,'Bin2',Bin2,'Bout',Bout);
            ks{end+1} = doubleLayer(convFFT(nImg,sK),convFFT(nImg,sK),'normLayer1',tvN,'normLayer2',tvN,'Bin1',Bin1,'Bin2',Bin2,'Bout',Bout);
            ks{end+1} = doubleLayer(convFFT(nImg,sK),convFFT(nImg,sK),'normLayer1',tvN,'normLayer2',tvN,'Bin1',Bin1,'Bin2',Bin2,'Bout',Bout,'storeInterm',1);
            
            ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'Bin2',randn(5,2));
            ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'Bin2',randn(5,2),'storeInterm',1);
            ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'Bin1',eye(24));
            ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'Bin1',eye(24),'storeInterm',1);
            ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'storeInterm',1);
            ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]));
            
            ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'Bin1',randn(24,3),'Bin2',randn(5,2),'Bout',randn(5,3));
            ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'Bin1',randn(24,3),'Bin2',randn(5,2),'Bout',randn(5,3),'storeInterm',1);
            
            
            testCase.layers = ks;
            
            
            
            
            % limitations on using anything with a dense layer and no 
            % reshape layer bc dimensions
            
%             ks    = cell(0,1);
%             TT = dense([14 14]);
%             ks{end+1} = doubleLayer(TT,TT);
%             tvNt  = tvNormLayer([2 7 14],'isWeight',1);
%             ks{end+1} = doubleLayer(dense([14 14]),dense([14 14]),'normLayer1',tvNt,'normLayer2',tvNt);
%             ks{end+1} = doubleLayer(dense([14 14]),dense([14 14]),'normLayer1',tvNt,'normLayer2',tvNt,'storeInterm',1);
%             tvNt  = tvNormLayer([2 7 14],'isWeight',0);
%             ks{end+1} = doubleLayer(dense([14 14]),dense([14 14]),'normLayer1',tvNt,'normLayer2',tvNt);
%             ks{end+1} = doubleLayer(dense([14 14]),dense([14 14]),'normLayer1',tvNt,'normLayer2',tvNt,'storeInterm',1);
%             ks{end+1} = doubleLayer(convFFT([12 8], [3 3 2 5]),dense([100,12*8*5]));
%             ks{end+1} = doubleLayer(convFFT([12 8], [3 3 2 5]),dense([100,12*8*5]),'storeInterm',1);
%             ks{end+1} = doubleLayer(dense([24 14]),dense([24 24]),'useGPU',0,'precision','single');
%             ks{end+1} = doubleLayer(dense([24 14]),dense([24 24]),'useGPU',0,'precision','single','storeInterm',1);
%             ks{end+1} = doubleLayer(dense([24 14]),dense([3 24]),'useGPU',0,'precision','single');
%             ks{end+1} = doubleLayer(dense([24 14]),dense([3 24]),'useGPU',0,'precision','single','storeInterm',1);
%             ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'useGPU',0,'precision','single');
%             ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'useGPU',0,'precision','single','storeInterm',1);
%             ks{end+1} = doubleLayer(convFFT([12 8], [3 3 2 5]),dense([100,12*8*5]),'useGPU',0,'precision','double');
%             ks{end+1} = doubleLayer(convFFT([12 8], [3 3 2 5]),dense([100,12*8*5]),'useGPU',0,'precision','double','storeInterm',1);
%             testCase.layers = ks;
        end
    end
end