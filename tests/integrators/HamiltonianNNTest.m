classdef HamiltonianNNTest < IntegratorTest
    % classdef ResNNTest < blockTest
    %
    % tests some Residual Nets. Extend to cover more cases.
    
    methods (TestClassSetup)
        function addIntegrators(testCase)
            ks    = cell(0,1);
            nc = 4;
            nt = 5;
            T = 1;
            K       = dense([nc,nc]);
            ks{end+1}  = HamiltonianNN(@tanhActivation,K,eye(nc),nt,T/nt);
%         
%             tA = linspace(0,T,nt); % array of time points
%         A = tA(:).^(0:3);
%             [Q,R] = qr(A);
%         
%                   K     = dense([4 4]);
%              layer = doubleSymLayer(K);
%               ks{end+1}   = DoubleHamiltonianNN(layer,layer,10,.1);

             %              ks{end+1}   = DoubleHamiltonianNN(layer,layer,100,.1);
            
           testCase.integrators = ks;
        end
    end
end