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
            testCase.integrators = ks;
        end
    end
end