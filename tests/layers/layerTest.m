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
                    th0 = initTheta(ks);
                    Y  = randn([sizeFeatIn(ks),10],'like',th0);
                    f = @(th) linearizeTheta(ks,th,Y);
                    isOK = checkJacobian(f,th0,'out',0,'useGPU',ks.useGPU,'precision',ks.precision);
                    testCase.verifyTrue(isOK);
                end
            end
            
        end
        
        function testLinearizeY(testCase)
            for k=1:numel(testCase.layers)
                ks = testCase.layers{k};
                th0 = initTheta(ks);
                Y  = randn([sizeFeatIn(ks),10],'like',th0);
                f = @(Y) linearizeY(ks,th0,Y);
                isOK = checkJacobian(f,Y,'out',0,'useGPU',ks.useGPU,'precision',ks.precision);
                
                testCase.verifyTrue(isOK);
            end
        end
        
        
        function testGetJThetaOp(testCase)
            for k=1:numel(testCase.layers)
                
                ks = testCase.layers{k};
                
                if nTheta(ks)>0
                    th0 = initTheta(ks);
                    Y  = randn([sizeFeatIn(ks),10],'like',th0);
                    [~,dA] = forwardProp(ks,th0,Y);
                    
                    J  = getJthetaOp(ks,th0,Y,dA);
                    [chkA,errA] = checkAdjoint(J);
                    if not(chkA)
                        fprintf('testGetJThetaOp: adjoint error = %1.2e\n',errA);
                    end
                    
                    testCase.verifyTrue(chkA);
                end
            end
        end
        function testGetJYOp(testCase)
            for k=1:numel(testCase.layers)
                ks = testCase.layers{k};
                
                th0 = initTheta(ks);
                Y  = randn([sizeFeatIn(ks),10],'like',th0);
                [~,dA] = forwardProp(ks,th0,Y);
                
                J  = getJYOp(ks,th0,Y,dA);
                [chkA,errA] = checkAdjoint(J,ks.useGPU,ks.precision);
                if not(chkA)
                    fprintf('testGetJYOp: adjoint error = %1.2e\n',errA);
                    
                end
                testCase.verifyTrue(chkA);
            end
        end
        
        function testGetJOp(testCase)
            for k=1:numel(testCase.layers)
                ks = testCase.layers{k};
                
                th0 = initTheta(ks);
                Y  = randn([sizeFeatIn(ks),10],'like',th0);
                [~,dA] = forwardProp(ks,th0,Y);
                
                J  = getJOp(ks,th0,Y,dA);
                [chkA,errA] = checkAdjoint(J,ks.useGPU,ks.precision);
                if not(chkA)
                    fprintf('testGetJOp: adjoint error = %1.2e\n',errA);
                end
                testCase.verifyTrue(chkA);
            end
        end
        
    end
end