clear, clc, close all

%Designparameter:
fs=12000; 
frametime=0.08; %zeitliche Aufl?sung in Sekunden
overlap=0.6; %?berlappung der frames



%einlesen und resampeln 6h_classic_piano
[y1,Fs1] = audioread('Bach2.mp3');
[y2,Fs2] = audioread('Bach3.mp3');
y1=sum(y1,2);
y1=resample(y1,fs,Fs1);
y2=sum(y2,2);
y2=resample(y2,fs,Fs2);
y=[y1;y2];


%STFT erzeugen
wlen = fs*frametime;
h = floor(wlen*(1-overlap));
nfft = wlen;
[stft, f, t] = stft(y, wlen, h, nfft, fs);
X=[real(stft);imag(stft)]; %transformiere in real und imag darstellung
featuredimensions=size(X)
% figure(1)
% imagesc(X)

csvwrite('FeaturesComplex_Bach.csv',X);
csvwrite('sizeFeaturesComplex_Bach.csv',size(X,1));

