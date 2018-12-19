classdef convMCNTest < kernelTest
	% classdef convMCNTest < kernelTest
	%
	% tests some convolutions with MatConvNet. Extend to cover more cases.
    methods (TestClassSetup)
        function addKernels(testCase)
            ks    = cell(0,1);
             ks{end+1} = convMCN([24 14],[3 3,3,4]);
            ks{end+1} = convMCN([14 24],[3 3,2,2]);
            ks{end+1} = convMCN([16 32],[3 3,2,2],'stride',2);
            ks{end+1} = convMCN([16 32],[1 1,2,4],'stride',2);
            ks{end+1} = convMCN([16 32],[2 2,3,4],'stride',2);
%             ks{end+1} = convMCN([16 32],[2 2,3,4],'pad',1);
%             ks{end+1} = convMCN([16 32],[2 2,3,4],'pad',[0 1 0 1]);
            ks{end+1} = convMCN([16 32],[3 3,3,4],'stride',2);
            ks{end+1} = convMCN([16 32],[3 3,2,2],'precision','single','Q',randn(3*3*2*2));
            ks{end+1} = convMCN([16 32],[3 3,2,2],'precision','single','Q',randn(3*3*2*2,10));
            testCase.kernels = ks;
        end
    end
end