% =========================================================================
% function P = getAveragePooling(nImg,np)
%
% constructs averages pooling operator
%
% Input:
%  nImg   - resolution of fine-scale image
%  np     - number of pooling steps (default: 1)
%
% Output:
%  P      - pooling operator (sparse matrix)
%
% =========================================================================
function P = getAveragePooling(nImg,np)

if nargin==0
    help(mfilename);
    runMinimalExample;
    return;
end

if not(exist('np','var')) || isempty(np)
    np = 1;
end

v = ones(1,2^np)/(2^np);

av = @(i) spdiags(ones(nImg(i),1)*v, 0:2^np-1, nImg(i),nImg(i));
A1    = av(1); A1 = A1(1:2^np:end,:) ;
A2    = av(2); A2 = A2(1:2^np:end,:) ;
Av    = kron(A2,A1);
P     = kron(speye(nImg(3)),Av);

function runMinimalExample
 I = double(imread('cameraman.tif'));
 n = size(I);
 figure(1); clf;
 subplot(2,3,1);
 imagesc(I);
 cax = caxis;
 title(sprintf('full resolution, n=[%d,%d]',n));
 for np=1:5
     P = feval(mfilename,[n 1],np);
     nc = n/(2^np);
     subplot(2,3,np+1);
     imagesc(reshape(P*I(:),nc));
     caxis(cax);
     title(sprintf('np=%d, n=[%d,%d]',np,nc));
 end
 
 