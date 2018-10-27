function L = getTVNormLayer(nData,varargin)
% constructs a getTVNormLayer for data of size nData. 
%
% Adds a scaling layer when isWeight==1
%
% Return type is a layer or NN.

% default parameter
isWeight  = 0; 
useGPU    = 0;
precision = 'double';

for k=1:2:length(varargin)     % overwrites default parameter
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end


if isWeight==1
    L    = tvNormLayer(nData,'useGPU',useGPU,'precision',precision);
else
    L = normLayer(nData,'doNorm',[0,1,0],'useGPU',useGPU,'precision',precision);
end