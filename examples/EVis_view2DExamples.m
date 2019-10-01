% visualizes the 2D examples used in 
%
% @article{HaberRuthotto2017,
%   author = {Haber, Eldad and Ruthotto, Lars},
%   title = {Stable architectures for deep neural networks} ,
%   journal = {Inverse Problems},
%   year = {2017},
%   volume = {34},
%   number = {1},
%   pages = {1--22},
% }


[Ye,Ce] = setupEllipses(800);

[Ys,Cs] = setupSwissRoll(800);

[Yp,Cp] = setupPeaks(800,5);

fig = figure(1); clf;
fig.Name = 'Meganet 2D datasets';
subplot(2,2,[1 2]);
viewFeatures2D(Ye,Ce)
axis equal tight
title('Ellipses')

subplot(2,2,3);
viewFeatures2D(Ys,Cs)
axis equal tight
title('Swiss Roll')

subplot(2,2,4);
viewFeatures2D(Yp,Cp)
axis equal tight
title('Peaks')
