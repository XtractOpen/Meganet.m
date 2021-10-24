classdef MegaNetTest < IntegratorTest
    % classdef MeganetTest < blockTest
    %
    % tests some Meganets. Extend to cover more cases.
    methods (TestClassSetup)
        function addBlocks(testCase)
            ks    = cell(1,1);
            blocks = cell(2,1);
            % first test
            blocks{1} = NN({singleLayer(dense([4 2]))});
            blocks{2} = ResNN(singleLayer(dense([4 4])),3,1);
            ks{1}    = Meganet(blocks);
            % ks{2}    = Meganet(blocks,'useGPU',0,'precision','single');
            
            convOp = @convMCN;
            convOpt = @convMCNt;
            act = @reluActivation;
            w = 8;
            d = 2; % latent space dimension
            
            % setup encoder
            enc1 = singleLayer(convOp([28 28 1],[4 4 1 w],'stride',2,'pad',1),'activation', act);
            enc2 = singleLayer(convOp([14 14 w],[4 4 w w*2],'stride',2,'pad',1),'activation', act);
            encr = reshapeLayer([7,7,w*2], w*2*7*7);
            enc3 = singleLayer(dense([d,w*2*7*7]),'activation',act);
            enc = NN({enc1,enc2,encr,enc3});
            
            
            % setup decoder
            dec0 = singleLayer(dense([w*2*7*7,d]),'activation',act);
            decr = reshapeLayer(w*2*7*7, [7,7,w*2]);
            dec1 = singleLayer(convOpt([14 14 w],[4 4 w w*2],'stride',2,'pad',1),'activation', act);
            dec2 = singleLayer(convOpt([28 28 1],[4 4 1 w],'stride',2,'pad',1),'activation', act);
            dec  = NN({dec0,decr,dec1,dec2});
            ks{end+1}  = Meganet({enc,dec});
            
            
            testCase.integrators = ks;
        end
    end
end