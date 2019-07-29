clear all; clc;
nImg = [16 16 4];
cout = 5;
Y = randn(nImg);

for sk=[1:5]
   for pad=0:floor(sk/2)
      for stride=1:2
         K = convMCN(nImg,[sk sk 4 cout],'pad',pad,'stride',stride);
         Kop = getOp(K,initTheta(K));
         Z = Kop*Y;
         assert(all(size(Z)==sizeFeatOut(K)),...
             sprintf('size(Z)=%s, sizeFeatOut=%s, sk=%d, pad=%d, stride=%d\n',...
                      num2str(size(Z)), num2str(sizeFeatOut(K)),sk, pad,stride));
         T = randn(size(Z));
         W = Kop'*T;
          t1 = T(:)'*Z(:);
          t2 = W(:)'*Y(:);
          assert(abs(t1-t2)/abs(t1) < 1e-10,...
              sprintf('adjoint test failed, sk=%d, pad=%d, stride=%d\n',...
                       sk, pad,stride))
      end
   end
end

