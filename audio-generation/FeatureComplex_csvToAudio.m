clear, clc, close all

%Designparameter:
fs=12000; 
frametime=0.08; %zeitliche Aufl?sung in Sekunden
overlap=0.6; %?berlappung der frames



%Lese csv-files
X=csvread('FeaturesComplex_Bach.csv');
S=size(X)
SIZE=csvread('sizeFeaturesComplex_Bach.csv');

if S(2)==1
    X=reshape(X,S(1)/SIZE,SIZE)';
    size(X)
end

wlen = fs*frametime;

%Spektrogramminversion
h = floor(wlen*(1-overlap));
nfft = wlen;
X=X(1:SIZE/2,:) + 1i*X(SIZE/2+1:end,:);
[x_istft, t_istft] = istft(X, h, nfft, fs);

% Spiele spektrogramminvertiertes Signal ab
sound(x_istft(1:100000),fs);