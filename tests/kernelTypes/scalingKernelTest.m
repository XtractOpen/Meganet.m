classdef scalingKernelTest < kernelTest
	% classdef scalingKernelTest < kernelTest
	%
	% tests some scaling kernels. Extend to cover more cases.
	methods (TestClassSetup)
        function addKernels(testCase)
            ks    = cell(0,1);
             ks{end+1} = scalingKernel([24,14,8],'isWeight',[1;0;0]);
             ks{end+1} = scalingKernel([24,14,8],'isWeight',[0;1;0],'precision','single');
             ks{end+1} = scalingKernel([24,14,8],'isWeight',[1;1;0]);
            testCase.kernels = ks;
        end
    end
end