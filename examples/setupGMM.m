function[Y,C,Yt,Ct] = setupGMM(dim,n_perClass,n_perClass_test,nc,GaussiansPerClass)

if not(exist('dim','var')) || isempty(dim)
    dim = 5;
end

if not(exist('n_perClass','var')) || isempty(n_perClass)
    n_perClass = 5000;
end

if not(exist('n_perClass_test','var')) || isempty(n_perClass_test)
    n_perClass_test = 1000;
end

if not(exist('nc','var')) || isempty(nc)
    nc = 5;
end

if not(exist('GaussiansPerClass','var')) || isempty(GaussiansPerClass)
    GaussiansPerClass = 5;
end

GMMweights = cell(nc);
GMMmeans = cell(nc,GaussiansPerClass);
GMMSigmaChol = cell(nc,GaussiansPerClass);
for c = 1:nc
    w = rand(GaussiansPerClass);
    GMMweights{c} = w/sum(w);
    for g = 1:GaussiansPerClass
        A = randn(dim,dim);
        A = 1000*inv(A'*A + 0.1*eye(dim));
        L = chol(A,'lower');
        GMMSigmaChol{c,g} = L;
        GMMmeans{c,g} = rand(dim,1); 
    end
end

%% samples

X = randn(dim,n_perClass*nc);
C = zeros(nc,n_perClass*nc);
Y = zeros(size(X));
gidx = 1; 
for c = 1:nc
    for g = 1:GaussiansPerClass
        n_Gaussian = round(GMMweights{c}(g)*n_perClass);
        endIdx = min(gidx+n_Gaussian-1,n_perClass*nc);
        if g==GaussiansPerClass && c==nc
            endIdx = n_perClass*nc;
        end
        nelem = endIdx - gidx + 1;
        Y(:,gidx:endIdx) = GMMSigmaChol{c,g}\(X(:,gidx:endIdx)) + repmat(GMMmeans{c,g},1,nelem);
        C(c , gidx:endIdx) = 1.0;
        gidx = gidx + nelem;
   end
end
if (sum(C(:)) ~= n_perClass*nc)
    sum(C(:))
    error('Try again');
end
if dim == 2
    viewFeatures2D(Y,C);
elseif dim == 3
    viewFeatures3D(Y,C);
end


Xt = randn(dim,n_perClass_test*nc);
Ct = zeros(nc,n_perClass_test*nc);
Yt = zeros(size(Xt));
gidx = 1; 
for c = 1:nc
    for g = 1:GaussiansPerClass
        n_Gaussian = round(GMMweights{c}(g)*n_perClass_test);
        endIdx = min(gidx+n_Gaussian-1,n_perClass_test*nc);
         if g==GaussiansPerClass && c==nc
            endIdx = n_perClass_test*nc;
        end
        nelem = endIdx - gidx + 1;
        Yt(:,gidx:endIdx) = GMMSigmaChol{c,g}\(X(:,gidx:endIdx)) + repmat(GMMmeans{c,g},1,nelem);
        Ct(c , gidx:endIdx) = 1.0;
        gidx = gidx + nelem;
   end
end
if (sum(Ct(:)) ~= n_perClass_test*nc)
    sum(Ct(:))
    error('Try again test');
end
if dim == 2
    viewFeatures2D(Y,C);
elseif dim == 3
    viewFeatures3D(Y,C);
end





end
