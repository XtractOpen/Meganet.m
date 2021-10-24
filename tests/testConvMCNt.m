clear all; clc;
nImg = [16 16 4];
cout = 5;

for sk=[1:5]
   for pad=0:floor(sk/2)
      for stride=1:2
         K = convMCNt(nImg,[sk sk nImg(3) cout],'pad',pad,'stride',stride);
         Y = randn(K.sizeFeatIn);
         th0 = initTheta(K);
         dth = randn(size(th0));
         Kop = getOp(K,th0);
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
         
         Zt = getOp(K,th0+dth)*Y;
         W = randn(size(Z));
         dZdth = K.JthetaTmv(Y,th0,W);
         f0 = dot(Zt(:),W(:));
         ft = dot(Z(:),W(:)) + dot(dth(:),dZdth(:));
         assert(abs(f0-ft)/abs(f0) < 1e-10,...
             sprintf('derivative test failed, sk=%d, pad=%d, stride=%d\n',...
             sk, pad,stride))
         
      end
   end
end

