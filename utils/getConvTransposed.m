function[KT] = getConvTransposed(K)

if nargin==0
    runMinimalExample
    return
end

sK = size(K);
KT = zeros([sK(1:2) sK(4) sK(3)],'like',K);
for m=1:size(K,3)
  for s = 1:size(K,4)
            Kms = K(:,:,m,s);
            KmsT = [Kms(3,3),Kms(3,2),Kms(3,1); ...
                    Kms(2,3),Kms(2,2),Kms(2,1); ...
                    Kms(1,3),Kms(1,2),Kms(1,1)]; 
            KT(:,:,s,m) = KmsT;
   end
end


function runMinimalExample;
sK = [3 3 4 6];
nImg = [8 8 6];
theta  = randn(sK);
K  = convFFT(nImg(1:2),sK);
Kt  = convFFT(nImg(1:2),[sK(1:2) sK(4) sK(3)]);

A  = getOp(K,theta);
rhs = randn(size(A,1),10);
th2  =feval(mfilename,theta);
At = getOp(Kt,th2);

t1 = A'*rhs;
t2 = At*rhs;
norm(t1(:)-t2(:))/norm(t1(:))