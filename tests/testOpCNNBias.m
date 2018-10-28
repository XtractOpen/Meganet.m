
nImg = [28 14 20];
Bop = opCNNBias(nImg);

th = 1:size(Bop,2);

Y = zeros([nImg 10]);
Z = Y + Bop*th;

for k=1:size(Bop,2)
    assert(all(vec(Z(:,:,k,:) == k)),'forward');
end


tt = Bop'*Z;
assert(all(tt==vec(prod([nImg(1:2) sizeLastDim(Y)])*(1:numel(th)))),'transpose');
