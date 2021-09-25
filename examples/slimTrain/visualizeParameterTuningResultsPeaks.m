

%% Allowable parameters

% Peaks
% batchSize    = [40,10,5,1];
% learningRate = [1e-4,1e-3,1e-2,1e-1];
% width        = [4,8,16];
% alphaW       = [1e-4,1e-3,1e-2,1e-1];
% memDepth     = [0,1,5,10];
% optMethod    = {'trialPoints','none'};



%% Select parameters

% among the best for ADAM for Peaks experiment
% myDirA = '~/Desktop/slimTikPeaksJul21/resultsPeaksADAM/';
myDirA = '~/Desktop/varproSAPaperResults/resultsPeaksADAM/';
bA    = [1];
lrA   = [1e-3];
wA    = [8];
aA    = [1e0];


myDirST = '~/Desktop/varproSAPaperResults/resultsPeaksSlimTik/';
bST    = [1,5,10];
lrST   = [1e-3];
wST    = [8];
aST    = [1e-10];
mdST   = [0,5,10];
mST    = {'trialPoints','constant'};



%% Load Results for ADAM
myFiles = dir(myDirA);

resultsADAM = {};
fctnADAM    = {};
namesADAM   = {};
thetaADAM   = {};
WADAM       = {};

count = 1;
for i = 1:length(myFiles)  
    if contains(myFiles(i).name,'.mat')
        [flag, fname] = findFiles(myFiles(i).name,wA,lrA,bA,aA);
        if flag
            tmp = load([myDirA,myFiles(i).name]);
            
            resultsADAM{count} = tmp.output.HIS;
            fctnADAM{count}    = tmp.output.fctn;
            namesADAM{count}   = fname;
            thetaADAM{count}   = tmp.output.thOpt;
            WADAM{count}       = tmp.output.WOpt;
            
            count = count + 1; 
        end
    end
end

%% Load Results for SlimTik
myFiles = dir(myDirST);

resultsSlimTik = {};
namesSlimTik   = {};
fctnSlimTik    = {};
thetaSlimTik   = {};
WSlimtTik      = {};

count = 1;
for i = 1:length(myFiles)  
    if contains(myFiles(i).name,'.mat')
        
        [flag, fname] = findFiles(myFiles(i).name,wST,lrST,bST,aST,mST,mdST);
        
        if flag
            tmp = load([myDirST,myFiles(i).name]);
            
            resultsSlimTik{count} = tmp.output.HIS;
            namesSlimTik{count}   = myFiles(i).name(31:end-4);
            fctnSlimTik{count}    = tmp.output.fctn;
            thetaSlimTik{count}   = tmp.output.thOpt;
            WSlimtTik{count}      = tmp.output.WOpt;
            
            
            count = count + 1; 
        end
    end
end


%% save information
saveDir = '~/Desktop/varproSAPaperResults/img/peaks/';

%% Plot Results for ADAM and SlimTik


fig = figure(3); clf;
fig.Name = 'ADAM vs. SlimTik Training';

colOrder = get(gca,'colororder');
colOrder = kron(colOrder,[1;1]);

idx = [];
for i = 1:length(resultsSlimTik)
    if contains(namesSlimTik{i},'sGCV')
        if contains(namesSlimTik{i},'constant')
            multilevelPlot(resultsSlimTik{i},'epoch','F','linear','log',0,{'--','LineWidth',3,'Color',colOrder(i,:)})
            idx = [idx,i];
        else
            multilevelPlot(resultsSlimTik{i},'epoch','F','linear','log',0,{'-','LineWidth',3,'Color',colOrder(i,:)})
            idx = [idx,i];
        end
    else
        continue
    end
        
    hold on;
end

for i = 1:length(resultsADAM)
    multilevelPlot(resultsADAM{i},'epoch','F','linear','log',0,{'k-o','LineWidth',3})
    hold on;
end


hold off;
% axis('off');
leg = legend(cat(2,namesSlimTik{idx},namesADAM),'Location','northoutside','interpreter','none','NumColumns',2,'FontSize',9);

ylim([1e-3,1e3])
ylabel('loss')
% matlab2tikz([saveDir,namesADAM{1},'.tex'],'width','\iwidth','height','\iheight');

return;
%% Display Alpha
fig = figure(4); clf;
fig.Name = 'Lambda per Iteration';

numEpochs = size(resultsSlimTik{1}{1}.his,1);

N = length(fctnSlimTik);

if ~exist('idx','var'), idx = 1:N; end
m = floor(sqrt(length(idx)));
n = ceil(length(idx) / m);


count = 0;
for i = 1:length(fctnSlimTik)
    figure(4); clf;
    % subplot(m,n,count+1);
    imagesc(log10(reshape(fctnSlimTik{i}.alphaWHist,[],numEpochs)));
    % colorbar;
%     xlabel('epoch');
%     ylabel('batch index');
    caxis([-12,0]);
    axis('square');
    set(gcf,'Color','w');
    set(gca,'FontSize',24);
    % title(namesSlimTik{i},'interpreter','none','FontSize',9);
    if contains(namesSlimTik{i},'trialPoints')
        export_fig([saveDir,'alpha',namesSlimTik{i},'.jpg']);
    end
    count = count + 1;
end

%% Display Approximations
fig = figure(5); clf;
fig.Name = 'Approx';


[X,Y] = meshgrid(linspace(-3,3,100),linspace(-3,3,100));
Z = peaks(X,Y);

imagesc(Z); 
set(gcf,'Color','w');
axis('square');
axis('off');
% export_fig([saveDir,'orig.jpg']);
cax = caxis;
caxDiff = [0,4];

fig1=figure;
% left=100; bottom=100 ; width=0; height=500;
% pos=[left bottom width height];
axis off
caxis([-12,0])
c = colorbar('FontSize',48);
set(fig1,'OuterPosition',pos)
set(gcf,'Color','w');
%set(c,'YTick',[]);
x1=get(gca,'position');
x=get(c,'Position');
x(3)=x(3)*0.5; % half the size
set(c,'Position',x)
set(gca,'position',x1)
% % % export_fig([saveDir,'colorbarApprox.jpg']);
% 
% fig1=figure;
% left=100; bottom=100 ; width=20 ; height=500;
% pos=[left bottom width height];
% axis off
% caxis(caxDiff)
% c = colorbar('FontSize',48);
% set(fig1,'OuterPosition',pos)
% set(gcf,'Color','w');
% % export_fig([saveDir,'colorbarDiff.jpg']);

% ADAM
YN = forwardProp(fctnADAM{1}.net,thetaADAM{1},[X(:)';Y(:)']);
ZN = reshape(WADAM{1},1,[]) * [YN;ones(1,size(YN,2))];
imagesc(reshape(ZN,size(X)));
caxis(cax);
set(gcf,'Color','w');
axis('square');
axis('off');
% export_fig([saveDir,'approx-',namesADAM{1},'.jpg']);

imagesc(abs(Z - reshape(ZN,size(X))));
set(gcf,'Color','w');
axis('square');
axis('off');
caxis(caxDiff);
% export_fig([saveDir,'diff-',namesADAM{1},'.jpg']);
fprintf([namesADAM{1},': %0.2e\n'],norm(Z(:) - ZN(:)) / norm(Z(:)))

N = length(fctnSlimTik);
m = floor(sqrt(N));
n = ceil(N / m);

for i = 1:N
    % subplot(m,n,i);
    YN = forwardProp(fctnSlimTik{i}.net,thetaSlimTik{i},[X(:)';Y(:)']);
    ZN = reshape(WSlimtTik{i},1,[]) * [YN;ones(1,size(YN,2))];
    imagesc(reshape(ZN,size(X)));
    set(gcf,'Color','w');
    axis('square');
    axis('off');
    % colorbar;
    caxis(cax);
    % title(namesSlimTik{i},'interpreter','none','FontSize',9);
    % export_fig([saveDir,'approx',namesSlimTik{i},'.jpg']);
end


fig = figure(6); clf;
fig.Name = 'Diff';


[X,Y] = meshgrid(linspace(-3,3,100),linspace(-3,3,100));
Z = peaks(X,Y);

N = length(fctnSlimTik);
m = floor(sqrt(N));
n = ceil(N / m);

for i = 1:N
    % subplot(m,n,i);
    YN = forwardProp(fctnSlimTik{i}.net,thetaSlimTik{i},[X(:)';Y(:)']);
    ZN = reshape(WSlimtTik{i},1,[]) * [YN;ones(1,size(YN,2))];
    imagesc(abs(Z - reshape(ZN,size(X))));
    set(gcf,'Color','w');
    axis('square');
    axis('off');
    % colorbar;
    caxis(caxDiff);
    fprintf([namesSlimTik{i},': %0.2e\n'],norm(Z(:) - ZN(:)) / norm(Z(:)))
    % title(sprintf('%0.2e',norm(Z(:) - ZN(:)) / norm(Z(:))),'interpreter','none','FontSize',9);
    % export_fig([saveDir,'diff',namesSlimTik{i},'.jpg']);
end


%% Helper Functions

function[] = multilevelPlot(HIS,xLabel,yLabel,xScale,yScale,valFlag,plotStyle)

if ~exist('valFlag','var'), valFlag = 0; end
if ~iscell(HIS), error('Not multilevel training results'); end


xIdx = find(strcmp(HIS{1}.str,xLabel));
yIdx = find(strcmp(HIS{1}.str,yLabel));

% concatenate
x = [];
y = [];
xStart = 0;
for i = 1:length(HIS)
    x = [x;xStart + HIS{i}.his(:,xIdx)];
    xStart = xStart + HIS{i}.his(end,xIdx);
    
    y = [y;HIS{i}.his(:,yIdx)];
end

h = plot(x,y(:,1),plotStyle{:});
if valFlag
    hold on;
    plot(x,y(:,2),'x-','LineWidth',2,'Color',h.Color,'HandleVisibility','off');
end


xlabel(xLabel);
ylabel(yLabel);

set(gca,'XScale',xScale);
set(gca,'YScale',yScale);
set(gca,'FontSize',18);
set(gcf,'Color','w');

grid on;

hold off;

end




function[flag,fname] = findFiles(fileName,width,lr,batch,alphaW,method,memDepth)

fname = '';

[flag,val] = check4Token(fileName,'width-%0.2d',width);
if flag == 0, return; end
fname = strcat(fname,sprintf('w-%0.2d',val));

[flag,val] = check4Token(fileName,'lr-%0.2e',lr);
if flag == 0, return; end
fname = strcat(fname,sprintf('-lr-%0.2e',val));

[flag,val] = check4Token(fileName,'batch-%0.2d',batch);
if flag == 0, return; end
fname = strcat(fname,sprintf('-b-%0.2d',val));

[flag,val] = check4Token(fileName,'alphaW-%0.2e',alphaW);
if flag == 0, return; end
fname = strcat(fname,sprintf('-aW-%0.2e',val));

if exist('method','var')
    [flag,val] = check4Token(fileName,'memDepth-%0.2d',memDepth);
    if flag == 0, return; end
    fname = strcat(fname,sprintf('-md-%0.2d',val));
    
    [flag,val] = check4Token(fileName,'method-%s',method);
    if flag == 0, return; end
    fname = strcat(fname,sprintf('-%s',val));
end

flag = 1;

end


function[flag,val] = check4Token(fileName,myStr,availOptions)

flag = 0;

for i = availOptions
        
    if iscell(i)
        val = i{1};
    else
        val = i;
    end
    
    substr = sprintf(myStr,val);
    
    if contains(fileName,substr)        
        flag = flag + 1;
        return;
    end
end

val = [];

end
