% =========================================================================
%
% Compute test results of trained CNNs. Evaluates loss and
% computes accuracies for test images. Also, appends results to the result
% file.
%
% References:
%
% Ruthotto L, Haber E: Deep Neural Networks motivated by PDEs,
%           arXiv:1804.04272 [cs.LG], 2018
%
% Haber E, Ruthotto L: Stable Architectures for Deep Neural Networks,
%      Inverse Problems, 2017
%
% Chang B, Meng L, Haber E, Ruthotto L, Begert D, Holtham E:
%      Reversible Architectures for Arbitrarily Deep Residual Neural Networks,
%      AAAI Conference on Artificial Intelligence 2018
%
% Inputs:
%
%     fname     - filename to load
%     ntest     - number of test images to analyze
%     useGPU    - flag for GPU computing, default = 0
%     precision - flag for precision, default = 'single'
%
% Output:
%
%     nex       - number of examples the network was trained on
%     hisTest   - loss, accuracy etc on test images
%
% Examples:
%
%     cnnResults();  runs minimal example
% =========================================================================

function [nex,hisTest] = cnnResults(fname,ntest,useGPU,precision)

if nargin==0
    feval(mfilename,'2018-11-24-02-20-07-cnnDriver-hamiltonian-cifar100.txt.mat',125,0,'single');
end


if not(exist('fname','var')) || isempty(fname)
    fname = '2018-11-24-02-20-07-cnnDriver-hamiltonian-cifar100.txt.mat';
end

if not(exist('useGPU','var')) || isempty(useGPU)
    useGPU = 0;
end

if not(exist('precision','var')) || isempty(precision)
    precision='single';
end

res = load(fname,'xc','ntrain','nval','nf','nt','idtrain','net','pAcc','PLast','CLast');
nex = numel(res.idtrain);
net = res.net;

% find out dynamic
if not(isempty(strfind(fname,'leapfrog')))
    dynamic = 'leapfrog';
    
elseif not(isempty(strfind(fname,'parabolic')))
    dynamic = 'parabolic';
    
elseif not(isempty(strfind(fname,'hamiltonian')))
    dynamic = 'hamiltonian';
elseif not(isempty(strfind(fname,'resnet')))
    dynamic = 'resnet';
    
    
else
    error('dynamic unknown %s',fname);
end


% theta = res.xOptAcc(1:nTheta(net));
% W = res.xOptAcc(nTheta(net)+1:end);
% [net,theta] = prolongateWeights(net,theta);
% res.xOptAcc= [theta;W];
% res.xc = res.xOptAcc;
% res.xOptLoss = res.xOptAcc;
% net = getCNN([nImg,cin],dynamic,res.nf,res.nt,useGPU,precision);

% res.net.blocks{1}.layers{1}.K.session=[];
fprintf('---- %s ntrain=%d -----\n',fname,nex)
    

% last iterate
if isfield(res,'pAcc') && isfield(res,'CLast') && isfield(res,'PLast') && size(res.PLast,2)==ntest && size(res.CLast,2)==ntest
    pAcc = res.pAcc;
    fprintf('load previous evaluations of test loss \n')
else
    
    % find out dataset
    if not(isempty(strfind(fname,'cifar100')))
        if not(exist('ntest','var')) || isempty(ntest)
            ntest = 10000;
        end
        dataset = 'cifar100';
        [~,~,Y0,C] = setupCIFAR100(0,ntest);
    elseif not(isempty(strfind(fname,'cifar10')))
        if not(exist('ntest','var')) || isempty(ntest)
            ntest = 10000;
        end
        dataset = 'cifar10';
        [~,~,Y0,C] = setupCIFAR10(0,ntest);
    elseif not(isempty(strfind(fname,'stl10')))
        if not(exist('ntest','var')) || isempty(ntest)
            ntest = 8000;
        end
        dataset = 'stl10';
        [~,~,Y0,C] = setupSTL(0,ntest);
    else
        error('dataset unknown %s',fname);
    end
    nImg   = [size(Y0,1) size(Y0,2)];
    cin    = size(Y0,3);
    Y0     = normalizeData(Y0,prod(nImg)*cin);
    id = randperm(ntest);
    Ytest = Y0(:,:,:,id);
    Ctest = C(:,id);
    
    [Ytest,Ctest] = gpuVar(useGPU,precision,Ytest,Ctest);
    ftest = dnnBatchObjFctn(net,[],softmaxLoss,[],Ytest,Ctest,'useGPU',useGPU,'precision',precision,'batchSize',125);
    
    [xc] = gpuVar(useGPU,precision,double(res.xc));
    
    hisTest = zeros(6,1);
    
    fprintf('evaluate  test loss \n')
    [Jc,p] = eval(ftest,xc);
    pAcc = gather(hisVals(ftest,p));
    hisTest(1:2) =[pAcc(1),pAcc(2)];
    [CLast,PLast] = getLabels(ftest,xc);
    CLast(:,id) = CLast;
    CLast = gather(CLast);
    PLast(:,id) = PLast;
    PLast = gather(PLast);
    save(fname,'CLast','PLast','pAcc','-append')
end
fprintf('loss=%1.4f accuracy=%1.4f\n',pAcc(1),pAcc(2));


