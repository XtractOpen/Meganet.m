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
            Qs = speye(numel(jval));
            ks{1} =  sparseKernel(ival,jval,nK);
            
             Qs = randn(numel(jval),2);
             ks{2} =  sparseKernel(ival,jval,nK,'Qs',Qs);
            
            testCase.kernels = ks;
        end
    end
end