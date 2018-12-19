classdef trafoTest < matlab.unittest.TestCase
    % main test file for trafos. specify all tests here and use specific
    % files to generate instances for particular kernel models.
    %
    properties
        trafos
    end
    
    methods (Test)
        
        function testLinearizeTheta(testCase)
           for k=1:numel(testCase.trafos)
              ks = testCase.trafos{k};
              Y  = randn([sizeFeatIn(ks),10]);
              f = @(th) linearizeTheta(ks,th,Y);
              th0 = randn(nTheta(ks),1);
              isOK = checkJacobian(f,th0);
              
              testCase.verifyTrue(isOK);
           end
        end

        function testLinearizeY(testCase)
           for k=1:numel(testCase.trafos)
              ks = testCase.trafos{k};
              theta = initTheta(ks);
              Y0  = randn([sizeFeatIn(ks),10]);
              f = @(Y) linearizeY(ks,theta,Y);
              isOK = checkJacobian(f,vec(Y0),'out',0);
              
              testCase.verifyTrue(isOK);
           end
        end

        function testVecInput(testCase)
        % test if kernel operates on vectorized and reshapes images alike
           for k=1:numel(testCase.trafos)
              ks = testCase.trafos{k};
              
              th = randn(nTheta(ks),1);
              Y  = randn([sizeFeatIn(ks),10]);
              Z  = ks.forwardProp(th,Y);
              
              Z1 = ks.forwardProp(th,vec(Y));
              Z2 = ks.forwardProp(th,reshape(Y,[sizeFeatIn(ks), 10]));
              
              testCase.verifyTrue(all(size(Z)==size(Z1)));
              testCase.verifyTrue(all(size(Z)==size(Z2)));
              
              testCase.verifyTrue(norm(Z(:)-Z1(:))/norm(Z(:)) < 1e-15);
              testCase.verifyTrue(norm(Z(:)-Z2(:))/norm(Z(:)) < 1e-15);
           end
        end            
        
        function testGetJThetaOp(testCase)
            for k=1:numel(testCase.trafos)
                ks = testCase.trafos{k};
                
                th = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                
                J  = getJthetaOp(ks,th,Y);
                chkA = checkAdjoint(J);
                testCase.verifyTrue(chkA);
            end
        end
        function testGetJYOp(testCase)
            for k=1:numel(testCase.trafos)
                ks = testCase.trafos{k};
                
                th = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                
                J  = getJYOp(ks,th,Y);
                chkA = checkAdjoint(J);
                testCase.verifyTrue(chkA);
            end
        end
        
        function testGetJOp(testCase)
            for k=1:numel(testCase.trafos)
                ks = testCase.trafos{k};
                
                th = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                
                J  = getJOp(ks,th,Y);
                chkA = checkAdjoint(J);
                testCase.verifyTrue(chkA);
            end
        end
         
    end
end