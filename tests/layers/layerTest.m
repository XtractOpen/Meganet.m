classdef layerTest < matlab.unittest.TestCase
    % main test file for layers. specify all tests here and use specific
    % files to generate instances for particular layers.
    %
    properties
        layers
    end
    
    methods (Test)
        
        function testLinearizeTheta(testCase)
            for k=1:numel(testCase.layers)
                ks = testCase.layers{k};
                if nTheta(ks)>0
                    Y  = randn([sizeFeatIn(ks),10]);
                    th0 = randn(nTheta(ks),1);
                    [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                    f = @(th) linearizeTheta(ks,th,Y);
                    isOK = checkJacobian(f,th0,'out',1,'useGPU',ks.useGPU,'precision',ks.precision);
                    
                    testCase.verifyTrue(isOK);
                end
            end
            
        end
        
        function testLinearizeY(testCase)
            for k=1:numel(testCase.layers)
                ks = testCase.layers{k};
                th0 = initTheta(ks);
                Y  = randn([sizeFeatIn(ks),10]);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                f = @(Y) linearizeY(ks,th0,Y);
                isOK = checkJacobian(f,vec(Y),'out',0,'useGPU',ks.useGPU,'precision',ks.precision);
                
                testCase.verifyTrue(isOK);
            end
        end
        
        function testVecInput(testCase)
            % test if kernel operates on vectorized and reshapes images alike
            for k=1:numel(testCase.layers)
                ks = testCase.layers{k};
                
                th = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                [Y,th] = gpuVar(ks.useGPU,ks.precision,Y,th);
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
            for k=1:numel(testCase.layers)
                ks = testCase.layers{k};
                
                th0 = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                [~,~,dA] = forwardProp(ks,th0,Y);
                
                J  = getJthetaOp(ks,th0,Y,dA);
                [chkA,errA] = checkAdjoint(J);
                testCase.verifyTrue(chkA);
            end
        end
        function testGetJYOp(testCase)
            for k=1:numel(testCase.layers)
                ks = testCase.layers{k};
                
                th0 = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                [~,~,dA] = forwardProp(ks,th0,Y);
                
                J  = getJYOp(ks,th0,Y,dA);
                [chkA,errA] = checkAdjoint(J,ks.useGPU,ks.precision);
                testCase.verifyTrue(chkA);
            end
        end
        
        function testGetJOp(testCase)
            for k=1:numel(testCase.layers)
                ks = testCase.layers{k};
                
                th0 = randn(nTheta(ks),1);
                Y  = randn([sizeFeatIn(ks),10]);
                [Y,th0] = gpuVar(ks.useGPU,ks.precision,Y,th0);
                [~,~,dA] = forwardProp(ks,th0,Y);
                
                J  = getJOp(ks,th0,Y,dA);
                [chkA,errA] = checkAdjoint(J,ks.useGPU,ks.precision);
                testCase.verifyTrue(chkA);
            end
        end
        
    end
end