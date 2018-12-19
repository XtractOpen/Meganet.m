classdef convFFTTest < kernelTest
	% classdef convFFTTest < kernelTest
	%
	% tests some convolutions with FFT. Extend to cover more cases.
	methods (TestClassSetup)
        function addKernels(testCase)
            ks    = cell(1,1);
            ks{1} = convFFT([24 14],[1 1,1,2]);
            ks{2} = convFFT([14 24],[5 5,2,2]);
            ks{3} = convFFT([14 24],[3 3,2,2],'useGPU',0);
            ks{4} = convFFT([14 24],[3 3,2,4],'useGPU',0,'precision','single');
            ks{4} = convFFT([14 24],[3 3,2,5],'precision','single');
            ks{4} = convFFT([14 24],[3 3,2,5],'precision','single','Q',randn(3*3*2*5,8));
            testCase.kernels = ks;
        end
    end
end