close all;
clear all;
clc;

data = csvread('data.csv');

[m,n] = size(data);

display(strcat('Read ',int2str(m),' records!'));

[coeff,score,latent,tsquared,explained,mu] = pca(data);

data = data*coeff;

display(explained);

figure;

[f,xi] = ksdensity(data(:,1));

plot(xi,f);

figure;

plot(data(:,1));