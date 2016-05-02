clear, clc, close all
%Designparameter:
% fs, scled linear
% wlen, beeinflusst Frequenzauflösung und somit die psychoakustische Redundanz
% h, der overlap scaled linear
% k, der Redundanzentferner scaled logarithmisch

%einlesen und resampeln
[y,Fs] = audioread('Bach1.mp3', [100000,700000]);
y=sum(y,2);
fs=2^14;
y=resample(y,fs,Fs);
%sound(y,fs);

%STFT erzeugen
wlen = 2^11;
h = floor(wlen/4);
nfft = wlen;
[stft, f, t] = stft(y, wlen, h, nfft, fs);

%Psychoakustische Redundanz entfernen
S=size(stft);
k=floor(wlen/10);%die Halbwertssamplelänge (mind. wlen/2)
rstft=reduce(stft,k);
Reduktionsfaktor=length(rstft(:,1))/S(1)

%Schreibe Feature in csv-file
csvwrite('features.csv',abs(rstft));

%Rekonstruiere stft
stft=reproduce(rstft,k,S(1));

%Spektrogramm erzeugen
MyDisplaySpectrogram(abs(stft), fs, h);

%Spektrogramminversion
[x_istft, t_istft] = istft(stft, h, nfft, fs);

%Spiele spektrogramminvertiertes Signal ab
sound(x_istft,fs);

