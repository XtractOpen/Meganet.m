classdef denseTest < kernelTest
	% classdef denseTest < kernelTest
	%
	% tests some dense kernels. Extend to cover more cases.
	methods (TestClassSetup)
        function addKernels(testCase)
            ks    = cell(1,1);
            ks{1} = dense([24 14]);
            ks{2} = dense([24 14],'precision','single');
            ks{3} = dense([24 14],'useGPU',0);
            ks{4} = dense([24 14],'useGPU',0,'precision','single');
            ks{5} = getDenseAntiSym([5,5]);
            testCase.kernels = ks;
        end
    end
end