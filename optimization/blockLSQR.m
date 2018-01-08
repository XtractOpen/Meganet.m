function[Xo,XoIter] = blockLSQR(A,B,iter,L,X)
%[W] = blockLSQR(X,Y)
%
if not(exist('iter','var')) || isempty(iter), iter = 150; end
if not(exist('L','var'))    || isempty(L), L = @(x) x; end
if not(exist('X','var')) || isempty(X),X = zeros(size(A,2),size(B,2));end

if isnumeric(L);  L = @(x) L\x;  end

Xin = X;
X   = 0*X;

Amv  = @(x) A*L(x);
Atmv = @(x) L(A'*x);

B = B-A*Xin;

if nargout>1
    XoIter = zeros(size(X,1),size(X,2),iter);
else
    XoIter = [];
end

beta  = norm(B,'fro');
U     = B/beta;
AtU   = Atmv(U);
alpha = norm(AtU,'fro');
V     = (AtU)/alpha;
W     = V; phiBar = beta; rhoBar = alpha;

plt = 0;

for i=1:iter
    Wc     = Amv(V) -alpha*U;
    beta   = norm(Wc,'fro');
    U      = Wc/beta;
    Sc     = Atmv(U) - beta*V;
    alpha  = norm(Sc,'fro');
    V      = Sc/alpha;
    rho    = sqrt(rhoBar^2 + beta^2);
    c      = rhoBar/rho;
    s      = beta/rho;
    theta  = s*alpha;
    rhoBar = c*alpha;
    phi    = c*phiBar;
    phiBar = -s*phiBar;
    X      = X + (phi/rho)*W;
    W      = V - (theta/rho)*W;
    
    if plt;fprintf('%3d   %3.2e\n',i,norm(Amv(X)-B,'fro')/norm(B,'fro'));end
    Xo = Xin + L(X);
    if nargout>1; XoIter(:,:,i) = Xin + Xo;end
    
    if plt
        AX = A*Xo;
        errRate(AX',B');
    end
    
end