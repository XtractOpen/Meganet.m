classdef reshapeLayerTest < layerTest
    % classdef singleLayerTest < layerTest
    %
    % tests some single layers. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            nImg = [32 32];
            nc = 8;
            ks{end+1} = reshapeLayer([nImg/8 nc],prod([nImg/8 nc]));            
            testCase.layers = ks;
        end
    end
end