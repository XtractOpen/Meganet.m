classdef kernelTest < matlab.unittest.TestCase
    % main test file for kernels. specify all tests here and use specific
    % files to generate instances for particular kernel models.
    %
    properties
        kernels
    end
    
    methods (Test)
        
        function tesFeatInOut(testCase)
           for k=1:numel(testCase.kernels)
              ks = testCase.kernels{k};
              
              th0 = randn(nTheta(ks),1);
              Y   = randn(nFeatIn(ks),10);
              [th0,Y] = gpuVar(ks.useGPU,ks.precision,th0,Y);
              Z   = getOp(ks,th0)*Y;
              Yt  = getOp(ks,th0)'*Z;
              
              
              testCase.verifyTrue(size(Z,1)==nFeatOut(ks));
              testCase.verifyTrue(size(Yt,1)==nFeatIn(ks));
           end
        end
        
        function testGPU(testCase)
            for k=1:numel(testCase.kernels)
              ks = testCase.kernels{k};
              if ks.useGPU == 1
                  
                  th   = randn(nTheta(ks),1);
                  Y0   = randn(nFeatIn(ks),10);
                  Z1   = getOp(ks,th)*Y0;
                  Y1   = getOp(ks,th)'*Z1;
                  
                  kg = ks;
                  kg.useGPU = 1;
                  [thg,Y0g] = gpuVar(kg.useGPU,kg.precision,th,Y0);
                  try 
                  Z1g   = getOp(kg,thg)*Y0g;
                  Y1g   = getOp(kg,thg)'*Z1g;
                  
                  err1 = norm(Z1(:)-vec(gather(Z1g)))/norm(Z1(:));
                  err2 = norm(Y1(:)-vec(gather(Y1g)))/norm(Y1(:));
                  
                  testCase.verifyTrue(err1 < 1e3*eps(thg(1)));
                  testCase.verifyTrue(err2 < 1e3*eps(thg(1)));
                  catch
                  end
              end

            end
        end            
            

        function testAdjoint(testCase)
            for k=1:numel(testCase.kernels)
              ks    = testCase.kernels{k};
              theta = randn(nTheta(ks),1);
              theta = gpuVar(ks.useGPU,ks.precision,theta);
              A     = getOp(ks,theta);
              OK    = checkAdjoint(A,ks.useGPU,ks.precision);
              testCase.verifyTrue(OK);
            end
        end

        function testDerivative(testCase)
        % test if kernel operates on vectorized and reshapes images alike
           for k=1:numel(testCase.kernels)
              ks = testCase.kernels{k};
              
              th  = randn(nTheta(ks),1);
              dth = randn(nTheta(ks),1);
              nex = 1;
              Y  = randn(nFeatIn(ks),nex)+nex;
              Z  = randn(nFeatOut(ks),nex)-nex;
              
              [th,Y,Z] = gpuVar(ks.useGPU, ks.precision,th,Y,Z);
              
              t1 = vec(Z)'*vec(Jthetamv(ks,dth,th,Y));
              t2 = vec(dth)'*vec(JthetaTmv(ks,Z,th,Y));
              testCase.verifyTrue(norm(t1-t2)/norm(t2) < 1e2*eps(gather(t1)));
           end
        end            
         
    end
end