classdef doubleLayerTest < layerTest
	% classdef doubleLayerTest < layerTest
    %
    % tests some double layers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            TT = dense([14 14]);
             ks{end+1} = doubleLayer(TT,TT);
              tvNt  = getTVNormLayer([2 7 14],'isWeight',1);        
               ks{end+1} = doubleLayer(dense([14 14]),dense([14 14]),'nLayer1',tvNt,'nLayer2',tvNt);
              tvNt  = getTVNormLayer([2 7 14],'isWeight',0);        
               ks{end+1} = doubleLayer(dense([14 14]),dense([14 14]),'nLayer1',tvNt,'nLayer2',tvNt);
               ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'Bin1',randn(24,3),'Bin2',randn(5,2),'Bout',randn(5,3));
               ks{end+1} = doubleLayer(convFFT([12 8], [3 3 2 5]),dense([100,12*8*5]));
                ks{end+1} = doubleLayer(dense([24 14]),dense([24 24]),'useGPU',0,'precision','single');
                ks{end+1} = doubleLayer(dense([24 14]),dense([3 24]),'useGPU',0,'precision','single');
                ks{end+1} = doubleLayer(dense([24 14]),dense([5 24]),'useGPU',0,'precision','single');
               ks{end+1} = doubleLayer(convFFT([12 8], [3 3 2 5]),dense([100,12*8*5]),'useGPU',0,'precision','double');
            testCase.layers = ks;
        end
    end
end