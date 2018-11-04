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
              Y  = randn([sizeFeatIn(ks),10]);
              if nTheta(ks)>0
              th0 = randn(nTheta(ks),1);
              [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
              f = @(th) linearizeTheta(ks,th,Y);
              isOK = checkJacobian(f,th0,'out',0,...
                                'useGPU',ks.useGPU,'precision',ks.precision);
              
              testCase.verifyTrue(isOK);
              end
           end
        end

        function testLinearizeY(testCase)
           for k=1:numel(testCase.integrators)
              ks = testCase.integrators{k};
              th0 = initTheta(ks);
              Y0  = randn([sizeFeatIn(ks),10]);
              [Y0,th0] = gpuVar(ks.useGPU,ks.precision,Y0,th0);
              f = @(Y) linearizeY(ks,th0,Y);
              isOK = checkJacobian(f,Y0,'out',0,...
                                'useGPU',ks.useGPU,'precision',ks.precision);
              testCase.verifyTrue(isOK);
           end
        end

               
        function testGetJThetaOp(testCase)
            for k=1:numel(testCase.integrators)
                ks = testCase.integrators{k};
                
                th0 = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                [~,dA] = forwardProp(ks,th0,Y);
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
                
                rng(1)
                th0 = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                [~,dA] = forwardProp(ks,th0,Y);
                
                J  = getJYOp(ks,th0,Y,dA);
                chkA = checkAdjoint(J,ks.useGPU,ks.precision);
                testCase.verifyTrue(chkA);
            end
        end
        
        function testGetJOp(testCase)
            for k=1:numel(testCase.integrators)
                ks = testCase.integrators{k};
                
                th0 = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                [~,dA] = forwardProp(ks,th0,Y);
                
                J  = getJOp(ks,th0,Y,dA);
                chkA = checkAdjoint(J,ks.useGPU,ks.precision);
                testCase.verifyTrue(chkA);
            end
        end
         
    end
end