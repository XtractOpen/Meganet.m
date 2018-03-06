function[X,Y] = sortAndScaleData(X,labels,option)
%[X,Y] = sortAndScaleData(X,labels)
%

% Scale X [-0.5 0.5]
%X  = X/max(abs(X(:))) - 0.5;
if nargin == 2, option = 1; end

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng
if option == 1
  X = bsxfun(@minus, X, mean(X,2)) ;
  n = std(X,0,2) ;
  X = bsxfun(@times, X, mean(n) ./ max(n, 40)) ;
end

if option == 2

  W = (X'*X)/size(X,1);
  [V,D] = eig(W);

  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  X = X*(V*diag(en./max(sqrt(d2), 10))*V');
end

if option == 3  
  X = bsxfun(@minus, X, mean(X,2)) ;
  X = X/200;
end


%% Organize labels
[~,k] = sort(labels);
labels = labels(k);
X      = X(k,:);

Y = zeros(size(X,1),max(labels)-min(labels)+1);
for i=1:size(X,1)
    Y(i,labels(i)+1) = 1;
end
