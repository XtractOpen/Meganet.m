classdef denseAntiSymTest < kernelTest
	% classdef denseAntiSymTest < kernelTest
	%
	% tests anti symmetric kernel. Extend to cover more cases.
    methods (TestClassSetup)
        function addKernels(testCase)
            ks    = cell(1,1);
            ks{1} = denseAntiSym([14 14]);
            ks{2} = denseAntiSym([14 14],'precision','single');
            ks{3} = denseAntiSym([14 14],'useGPU',1);
            ks{4} = denseAntiSym([14 14],'useGPU',1,'precision','single');
            testCase.kernels = ks;
        end
    end
end