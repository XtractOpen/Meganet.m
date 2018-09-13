function [ConfMat] = getConfusionMatrix(Cp,Ctrue)
%[ConfMat] = getConfusionMatrix(Cp,Ctrue)
%
% computes the confusion matrix between predicted labels and true labels.
%
% Inputs:
%   Cp     -  predicted labels
%   Ctrue  -  true labels
%
% Outputs:
%   ConfMat - confusion matrix

if nargin==0; help(mfilename); runMinimalExample; return; end

nc = size(Ctrue,1);  % number of classes
nex = size(Ctrue,2); % number of examples

ConfMat = zeros(nc);
for i=1:nex
    j1 = find(Cp(:,i));
    j2 = find(Ctrue(:,i));
    
    ConfMat(j1,j2) = ConfMat(j1,j2)+1;
end

function runMinimalExample
nc  = 6;
nex = 100;

Ptrue = rand(nc,nex); Ptrue = Ptrue./sum(Ptrue,1);
Pp    = rand(nc,nex); Pp = Pp./sum(Pp,1);

[~,ct] = max(Ptrue,[],1);
Ctrue = full(sparse(ct,1:nex,ones(nex,1),nc,nex));

[~,cp] = max(Pp,[],1);
Cp = full(sparse(cp,1:nex,ones(nex,1),nc,nex));

ConfMat = feval(mfilename,Cp,Ctrue);
figure(); clf;
imagesc(ConfMat)



