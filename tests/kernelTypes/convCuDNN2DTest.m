classdef convCuDNN2DTest < kernelTest
	% classdef convMCNTest < kernelTest
	% tests some convolutions with cuDNN. Extend to cover more cases.
    methods (TestClassSetup)
        function addKernels(testCase)
            ks    = cell(0,0);
            ks{end+1} = convCuDNN2D([],[24 14],[3 3,1,2]);
            ks{end+1} = convCuDNN2D(convCuDNN2DSession(),[24 14],[3 3,1,2]);
            ks{end+1} = convCuDNN2D(convCuDNN2DSession(),[14 24],[3 3,2,2]);
            ks{end+1} = convCuDNN2D([],[14 24],[3 3,2,2]);
            ks{end+1} = convCuDNN2D([],[16 32],[3 3,2,2]);
            ks{end+1} = convCuDNN2D(convCuDNN2DSession(),[16 32],[3,3,1,4],'stride',2);
            ks{end+1} = convCuDNN2D(convCuDNN2DSession(),[16 32],[3,3,1,4],'Q',randn(3*3*1*4,3));

            testCase.kernels = ks;
        end
    end
end