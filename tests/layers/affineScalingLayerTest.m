classdef affineScalingLayerTest < layerTest
	% classdef affineScalingLayerTest < layerTest
	%
	% tests instance affineScalingLayerTest. Extend to cover more cases.
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            ks{end+1} = affineScalingLayer([8 4 23],'isWeight',[1,0,0]);
            ks{end+1} = affineScalingLayer([8 4 23],'isWeight',[0,1,0]);
            ks{end+1} = affineScalingLayer([8 4 23],'isWeight',[1,1,0]);
            ks{end+1} = affineScalingLayer([8 4 23],'isWeight',[1,1,0],'useGPU',1);
            ks{end+1} = affineScalingLayer([8 4 23],'isWeight',[1,1,0],'useGPU',1,'precision','single');
            testCase.layers = ks;
        end
    end
    
end