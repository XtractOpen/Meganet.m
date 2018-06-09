function [X] = normalizeData(X)
% image normalization as described in
% https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
%
m=mean(X,2);
X=bsxfun(@minus, X, m);
s=std(X,[],2);
X=bsxfun(@rdivide, X, max(s,1/sqrt(size(X,2))));
end
