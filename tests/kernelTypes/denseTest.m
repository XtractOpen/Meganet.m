classdef denseTest < kernelTest
	% classdef denseTest < kernelTest
	%
	% tests some dense kernels. Extend to cover more cases.
	methods (TestClassSetup)
        function addKernels(testCase)
            ks    = cell(0,1);
            ks{end+1} = dense([24 14]);
            ks{end+1} = dense([24 14],'precision','single');
            ks{end+1} = dense([24 14],'useGPU',0);
            ks{end+1} = dense([24 14],'useGPU',0,'precision','single');
            ks{end+1} = getDenseAntiSym([5,5]);
            testCase.kernels = ks;
        end
    end
end