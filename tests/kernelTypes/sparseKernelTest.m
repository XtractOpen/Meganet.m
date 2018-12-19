classdef sparseKernelTest < kernelTest
	% classdef sparseKernelTest < kernelTest
	%
	% tests some dense kernels. Extend to cover more cases.
	methods (TestClassSetup)
        function addKernels(testCase)
            ks    = cell(1,1);
            nK    = [100,200];
            tmp   = sprandn(nK(1),nK(2),0.1);
            [ival,jval,~] =  find(tmp);
            ks{1} =  sparseKernel(ival,jval,nK);
            
             ks{2} =  sparseKernel(ival,jval,nK);
            
            testCase.kernels = ks;
        end
    end
end