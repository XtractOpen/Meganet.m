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
              Y   = randn([sizeFeatIn(ks),10]);
              [th0,Y] = gpuVar(ks.useGPU,ks.precision,th0,Y);
              Z   = getOp(ks,th0)*Y;
              Yt  = getOp(ks,th0)'*Z;
              
%               colonsZ = repmat( {':'} , 1 , ndims(Z) -1 );
%               colonsYt = repmat( {':'} , 1 , ndims(Yt) -1 );
%               testCase.verifyTrue( all( size( Z(colonsZ{:} ,1))==sizeFeatOut(ks)) ); % size() squeezes out the trailing 1's
%               testCase.verifyTrue( all( size(Yt(colonsYt{:},1))==sizeFeatIn(ks) ) );
              szZ = size(Z);
              szYt = size(Yt);
              testCase.verifyTrue( all( szZ(1:end-1)==sizeFeatOut(ks)) ); % TODO still may fail if nex=1
              testCase.verifyTrue( all( szYt(1:end-1)==sizeFeatIn(ks)) );
           end
        end
        
        function testGPU(testCase)
            for k=1:numel(testCase.kernels)
              ks = testCase.kernels{k};
              if ks.useGPU == 1
                  
                  th   = initTheta(ks);
                  Y0   = randn([sizeFeatIn(ks),10],'like',th);
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
              theta = initTheta(ks);
              A     = getOp(ks,theta);
              OK    = checkAdjoint(A,ks.useGPU,ks.precision);
              testCase.verifyTrue(OK);
            end
        end

        function testDerivative(testCase)
        % test if kernel operates on vectorized and reshapes images alike
           for k=1:numel(testCase.kernels)
              ks = testCase.kernels{k};
              
              th  = initTheta(ks);
              if isprop(ks,'sK')
                dth = randn(ks.sK,'like', th); 
              else
                dth = randn(nTheta(ks),1,'like',th); % dense kernel has no sK
              end
              nex = 1;
              Y  = randn([sizeFeatIn(ks),nex],'like',th)+nex;
              Z  = randn([sizeFeatOut(ks),nex],'like',th)-nex;
              
              [th,Y,Z] = gpuVar(ks.useGPU, ks.precision,th,Y,Z);
              
              t1 = sum(vec(Z.* Jthetamv(ks,dth,th,Y)));
              t2 = sum(vec(dth.*JthetaTmv(ks,Z,th,Y)));
              testCase.verifyTrue(norm(t1-t2)/norm(t2) < 1e2*eps(gather(t1)));
           end
        end            
         
    end
end