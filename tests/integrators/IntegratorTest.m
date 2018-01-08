classdef IntegratorTest < matlab.unittest.TestCase
    % main test file for integrators. specify all tests here and use specific
    % files to generate instances for particular kernel models.
    %
    properties
        integrators
    end
    
    methods (Test)

        function testLinearizeTheta(testCase)
           for k=1:numel(testCase.integrators)
              ks = testCase.integrators{k};
              Y  = randn(nFeatIn(ks),10);
              if nTheta(ks)>0
              th0 = randn(nTheta(ks),1);
              [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
              f = @(th) linearizeTheta(ks,th,Y);
              isOK = checkJacobian(f,th0,'out',1,...
                                'useGPU',ks.useGPU,'precision',ks.precision);
              
              testCase.verifyTrue(isOK);
              end
           end
        end

        function testLinearizeY(testCase)
           for k=1:numel(testCase.integrators)
              ks = testCase.integrators{k};
              th0 = initTheta(ks);
              Y0  = randn(nFeatIn(ks),10);
              [Y0,th0] = gpuVar(ks.useGPU,ks.precision,Y0,th0);
              f = @(Y) linearizeY(ks,th0,Y);
              isOK = checkJacobian(f,vec(Y0),'out',0,...
                                'useGPU',ks.useGPU,'precision',ks.precision);
              testCase.verifyTrue(isOK);
           end
        end

        function testVecInput(testCase)
        % test if kernel operates on vectorized and reshapes images alike
           for k=1:numel(testCase.integrators)
              ks = testCase.integrators{k};
              
              th0 = randn(nTheta(ks),1);
              Y  = randn(nFeatIn(ks),10);
              [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
              [~,Z]  = ks.apply(th0,Y);
              
              [~,Z1] = ks.apply(th0,vec(Y));
              [~,Z2] = ks.apply(th0,reshape(Y,[nFeatIn(ks), 10]));
              
              testCase.verifyTrue(all(size(Z)==size(Z1)));
              testCase.verifyTrue(all(size(Z)==size(Z2)));
              
              if isempty(th0)||numel(th0)==0
                  break;
              end
                testCase.verifyTrue(norm(Z(:)-Z1(:))/norm(Z(:)) < 1e2*eps(th0(1)));
                testCase.verifyTrue(norm(Z(:)-Z2(:))/norm(Z(:)) < 1e2*eps(th0(1)));
           end
        end            
        
        function testGetJThetaOp(testCase)
            for k=1:numel(testCase.integrators)
                ks = testCase.integrators{k};
                
                th0 = randn(nTheta(ks),1);
                Y  = randn(nFeatIn(ks),10);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                [~,~,dA] = apply(ks,th0,Y);
                if nTheta(ks)>0
                J  = getJthetaOp(ks,th0,Y,dA);
                chkA = checkAdjoint(J,ks.useGPU,ks.precision);
                testCase.verifyTrue(chkA);
                end
            end
        end
        function testGetJYOp(testCase)
            for k=1:numel(testCase.integrators)
                ks = testCase.integrators{k};
                
                th0 = randn(nTheta(ks),1);
                Y  = randn(nFeatIn(ks),10);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                [~,~,dA] = apply(ks,th0,Y);
                
                J  = getJYOp(ks,th0,Y,dA);
                chkA = checkAdjoint(J,ks.useGPU,ks.precision);
                testCase.verifyTrue(chkA);
            end
        end
        
        function testGetJOp(testCase)
            for k=1:numel(testCase.integrators)
                ks = testCase.integrators{k};
                
                th0 = randn(nTheta(ks),1);
                Y  = randn(nFeatIn(ks),10);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                [~,~,dA] = apply(ks,th0,Y);
                
                J  = getJOp(ks,th0,Y,dA);
                chkA = checkAdjoint(J,ks.useGPU,ks.precision);
                testCase.verifyTrue(chkA);
            end
        end
         
    end
end