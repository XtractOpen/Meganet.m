clear all; clc;
sK   = [3 3 2 3];
th   = randn(sK);
nImg = [48 48];
Kmat = getConvCoupleMat(th,[nImg sK(3)]);

Y = randn([nImg,sK(3)]);
Z = reshape(Kmat*Y(:), [nImg sK(4)]);

err = @(X,Y) norm(vec(X(2:end-1,2:end-1,:)-Y(2:end-1,2:end-1,:)))/norm(vec(Y(2:end-1,2:end-1,:)));

useGPU = [0 ];
precision = {'single','double'};

for g = 1:numel(useGPU)
    for p=1:numel(precision)
       fprintf('--- comparing convolutions (%s, useGPU:%d) ---\n',precision{p},useGPU(g));
       
       [Yt,tht] = gpuVar(useGPU(g),precision{p},Y,th);
       
       % use convMCN
       Kt = convMCN(nImg,sK,'useGPU',useGPU(g),'precision',precision{p});
       Zt = reshape(getOp(Kt,tht)*Yt,[nImg sK(4)]);
       fprintf('\tconvMCN: %1.2e\n',err(gather(Zt),Z));
       % use convFFT
       Kt = convFFT(nImg,sK,'useGPU',useGPU(g),'precision',precision{p});
       Zt = reshape(getOp(Kt,tht)*Yt,[nImg sK(4)]);
       fprintf('\tconvFFT: %1.2e\n',err(gather(Zt),Z));
       % use convCUDNN
       if useGPU(g)==1 && strcmp(precision{p},'single')
           cudnnSession = convCuDNN2DSession();
           Kt = convCuDNN2D(cudnnSession,nImg,sK,'useGPU',useGPU(g),'precision',precision{p});
           Zt = reshape(getOp(Kt,tht)*Yt,[nImg sK(4)]);
           fprintf('\tconvCUDNN: %1.2e\n',err(gather(Zt),Z));
       end
    end
end

