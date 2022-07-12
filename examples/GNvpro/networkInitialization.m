
function[theta,W] = networkInitialization(net, method, seed, numClasses, addBias)


if ~exist('seed','var') || isempty(seed), seed = 20; end
rng(seed);

if ~exist('numClasses','var'), numClasses = []; W = []; end
if ~exist('addBias','var'), addBias = 1; end

switch method
    
    case 'xavier'
        % http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        theta = [];
        for i = 1:length(net.blocks)
            n = nTheta(net.blocks{i});
            
            nIn  = prod(sizeFeatIn(net.blocks{i}));
            nOut = prod(sizeFeatOut(net.blocks{i}));
            theta = cat(1,theta,(sqrt(6) / sqrt(nIn + nOut)) * (rand(n,1) - 0.5));
        end
        
    case 'kaiming_uniform'
        % https://arxiv.org/pdf/1502.01852.pdf
        theta = [];
        for i = 1:length(net.blocks)
            if nTheta(net.blocks{i}) > 0
                for j = 1:length(net.blocks{i}.layers)
                    
                    [fanIn,~] = calculateFanInFanOut(net.blocks{i}.layers{j}.K.sK);
                    
                    nK = prod(net.blocks{i}.layers{j}.K.sK);
                    
                    % weights
                    a     = sqrt(5);                % negative slope?
                    gain  = sqrt(2 / (1 + a^2));    % leaky relu gain
                    b     = gain * sqrt(3 / fanIn);  % bound
                    theta = cat(1,theta,(2 * b) * (rand(nK,1) - 0.5));
                    
                    % bias
                    nB    = numel(net.blocks{i}.layers{j}.Bin) + numel(net.blocks{i}.layers{j}.Bout);
                    b     = 1 / sqrt(fanIn);
                    theta = cat(1,theta,(2 * b) * (rand(nB,1) - 0.5));
                end
            end
        end
        
        if ~isempty(numClasses)
            % fan in method
            nW    = numClasses * (prod(sizeFeatOut(net)));
            nB    = numClasses * addBias;
            fanIn = prod(sizeFeatOut(net));
            a     = sqrt(5);                % negative slope?
            gain  = sqrt(2 / (1 + a^2));    % leaky relu gain
            b     = gain * sqrt(3 / fanIn);  % bound
            W     = (2 * b) * (rand(nW,1) - 0.5);
            
            % bias
            b = 1 / sqrt(fanIn);
            W = cat(1,W,(2 * b) * (rand(nB,1) - 0.5));
            
        end


    otherwise
        theta =  1e-3*vec(randn(nTheta(net),1));
end

end



function[fanIn,fanOut] = calculateFanInFanOut(sK)

if length(sK) == 2
    % fully connected
    fanIn  = sK(2); 
    fanOut = sK(1);
else
    % convolution
    filterSize = sK(1) * sK(2);
    
    fanIn  = sK(end-1) * filterSize;
    fanOut = sK(end) * filterSize;
    
end

end
