

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
myDirA = '~/Desktop/varproSAPaperResults/resultsCDRADAM/';
bA    = [5];
lrA   = [1e-3,1e-2,1e-1];
wA    = [16];
aA    = [1e-10];


myDirST = '~/Desktop/varproSAPaperResults/resultsCDRSlimTik/';
bST    = [5];
lrST   = [1e-3,1e-2,1e-1];
wST    = [16];
aST    = [1e-10];
mdST   = [0,5,10];
mST    = {'trialPoints'};



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
%             
%             disp(fname)
%             x = tmp.output.info.relErrTrainOpt;
%             fprintf('TRAIN\t$%0.3f \\pm %0.3f$ & $%0.3f$& $%0.3f$\n',mean(x),std(x),min(x),max(x));
%             x = tmp.output.info.relErrValOpt;
%             fprintf('VAL\t$%0.3f \\pm %0.3f$& $%0.3f$& $%0.3f$\n',mean(x),std(x),min(x),max(x));
%             x = tmp.output.info.relErrTestOpt;
%             fprintf('TEST\t$%0.3f \\pm %0.3f$& $%0.3f$& $%0.3f$\n',mean(x),std(x),min(x),max(x));

            resultsADAM{count} = tmp.output.HIS;
            
            fprintf('%s:\t%0.3f\t%0.3f\n',fname,resultsADAM{count}{1}.his(20,7),resultsADAM{count}{1}.his(20,end-1));
            
            fctnADAM{count}    = tmp.output.fctn;
            namesADAM{count}   = fname;
            thetaADAM{count}   = tmp.output.thOpt;
            WADAM{count}       = tmp.output.WOpt;
%             
%             f = tmp.output.fctn;
%             f.Y = Yt; f.C = Ct; 
%             Jct = eval(f,[thetaADAM{count}(:);WADAM{count}(:)]);
%             
%             f.Y = Yv; f.C = Cv;
%             Jcv = eval(f,[thetaADAM{count}(:);WADAM{count}(:)]);
%             
%             f.Y = Ytest; f.C = Ctest;
%             Jctest = eval(f,[thetaADAM{count}(:);WADAM{count}(:)]);
            
%            fprintf('%s:\t%0.2e\t%0.2e\t%0.2e\n',fname,Jct,Jcv,Jctest);
            
%             disp(namesADAM{count})
%             disp(min(resultsADAM{count}{1}.his(:,end-1)))
            
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

rng(20);
[Yt,Ct,Yv,Cv,Ytest,Ctest,idx] = setupCDR(400,200);

count = 1;
for i = 1:length(myFiles)  
    if contains(myFiles(i).name,'.mat')
        
        [flag, fname] = findFiles(myFiles(i).name,wST,lrST,bST,aST,mST,mdST);
        
        if flag
            tmp = load([myDirST,myFiles(i).name]);
            
%             disp(fname)
%             x = tmp.output.info.relErrTrainOpt;
%             fprintf('TRAIN\t$%0.3f \\pm %0.3f$ & $%0.3f$& $%0.3f$\n',mean(x),std(x),min(x),max(x));
%             x = tmp.output.info.relErrValOpt;
%             fprintf('VAL\t$%0.3f \\pm %0.3f$& $%0.3f$& $%0.3f$\n',mean(x),std(x),min(x),max(x));
%             x = tmp.output.info.relErrTestOpt;
%             fprintf('TEST\t$%0.3f \\pm %0.3f$& $%0.3f$& $%0.3f$\n',mean(x),std(x),min(x),max(x));
            
            resultsSlimTik{count} = tmp.output.HIS;
            fprintf('%s:\t%0.3f\t%0.3f\n',fname,resultsSlimTik{count}{1}.his(20,7),resultsSlimTik{count}{1}.his(20,end-1));
           
            
            namesSlimTik{count}   = myFiles(i).name(31:end-4);
            fctnSlimTik{count}    = tmp.output.fctn;
            thetaSlimTik{count}   = tmp.output.thOpt;
            WSlimtTik{count}      = tmp.output.WOpt;
            

%             disp(namesSlimTik{count})
%             disp(min(resultsSlimTik{count}{1}.his(:,end-1)))

%             f = tmp.output.fctn;
%             f.WPrev = WSlimtTik{count};
%             f.Y = Yt; f.C = Ct; 
%             Jct = eval(f,thetaSlimTik{count});
%             
%             f.Y = Yv; f.C = Cv;
%             Jcv = eval(f,thetaSlimTik{count});
%             
%             f.Y = Ytest; f.C = Ctest;
%             Jctest = eval(f,thetaSlimTik{count});
            
%            fprintf('%s:\t%0.2e\t%0.2e\t%0.2e\n',fname,Jct,Jcv,Jctest);

            

%             % print out error
%             disp(namesSlimTik{count});
%             zn = eval(tmp.output.net,thetaSlimTik{count},Yt);
%             cn = reshape(WSlimtTik{count},size(Ct,1),[]) * [zn;ones(1,size(zn,2))];
%             
            


            count = count + 1; 
        end
    end
end

return;
%% save information
saveDir = '~/Desktop/varproSAPaperResults/img/cdr/';



%% Plot Results for ADAM and SlimTik


fig = figure(3); clf;
fig.Name = 'ADAM vs. SlimTik Training';

colOrder = get(gca,'colororder');
colOrder = kron(colOrder,[1;1]);

idx = [];
for i = 1:length(resultsSlimTik)
    if contains(namesSlimTik{i},'sGCV')
        if contains(namesSlimTik{i},'constant')
            % multilevelPlot(resultsSlimTik{i},'epoch','F','linear','log',1,{'--','LineWidth',3,'Color',colOrder(i,:)})
            % idx = [idx,i];
        else
            if contains(namesSlimTik{i},'memDepth-00')
                multilevelPlot(resultsSlimTik{i},'epoch','F','linear','log',1,{'-','LineWidth',3,'Color',colOrder(i,:)})
            end
            idx = [idx,i];
        end
    else
        continue
    end
        
    hold on;
end

for i = 1:length(resultsADAM)
    multilevelPlot(resultsADAM{i},'epoch','F','linear','log',1,{'k-o','LineWidth',3})
    hold on;
end


hold off;
% leg = legend(cat(2,namesSlimTik{idx},namesADAM),'Location','northoutside','interpreter','none','NumColumns',2,'FontSize',9);

ylim([1e0,1e5])
ylabel('loss')
% matlab2tikz([saveDir,'best-with-validation.tex'],'width','\iwidth','height','\iheight');

%% Plot |theta1 - theta0| for various learning rates

fig = figure(2); clf;
for i = 1:length(resultsSlimTik)
    if contains(namesSlimTik{i},'trialPoints')
        semilogy(resultsSlimTik{i}{1}.his(1:end,3),'LineWidth',3);
        hold on;
    end
end
xlabel('epoch')
ylabel('$\|\mathbf{\theta}_{k} - \mathbf{\theta}_{k-1}\|$','interpreter','latex')

set(gca,'FontSize',18);
set(gcf,'Color','w');

hold off;

% matlab2tikz([saveDir,'thetaChangePerLearningRate_r10.tex'],'width','\iwidth','height','\iheight');

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
        export_fig([saveDir,'alpha-w',namesSlimTik{i},'.jpg']);
    end
    count = count + 1;
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
