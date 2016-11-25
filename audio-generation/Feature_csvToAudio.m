clear, clc, close all

%Designparameter:
fs=12000; 
frametime=0.08; %zeitliche Aufl?sung in Sekunden
overlap=0.6; %?berlappung der frames
p=2; %psychoakustische redundanz (je grösser desto mehr reduktion)
red=0.8; %pvd reduktion (je kleiner desto mehr reduktion)

%dimensionalityreduction (setze 1 für Aktivierung)
psychoakustik=1;
SVD=0;


%Lese csv-files
X=csvread('features.csv')';
S=size(X)
SIZE=csvread('sizeFaturesAmplitude_Bach.csv');
if S(2)==1
    X=reshape(X,S(1)/SIZE(2),SIZE(2))';
    size(X)
end

figure(1)
imagesc(X)


%SVD backtrafo
if(SVD==1)
    Vr=csvread('svd_backtrafo.csv');
    X=(X'*Vr)'; %the backtransformed featurematrix
end

%psychoakustische backtrafo
wlen = fs*frametime;
if psychoakustik==1
    k=floor(wlen/(p*10));
    X=reproduce(X,k,SIZE(1));    
end
figure(2)
imagesc(X)

%Spektrogramminversion
h = floor(wlen*(1-overlap));
nfft = wlen;
[x_istft, t_istft] = istft(X, h, nfft, fs);

% Spiele spektrogramminvertiertes Signal ab
sound(x_istft(1:100000),fs);
