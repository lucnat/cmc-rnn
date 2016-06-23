clear, clc, close all
%Designparameter:
% fs, scled linear
% wlen, beeinflusst Frequenzaufl?sung und somit die psychoakustische Redundanz
% h, der overlap scaled linear
% k, der Redundanzentferner scaled logarithmisch

%einlesen und resampeln
[y,Fs] = audioread('Bach1.mp3');
y=sum(y,2);
fs=3000;
y=resample(y,fs,Fs);
%sound(y,fs);

%STFT erzeugen
frametime=0.1; %zeitliche Aufl?sung in Sekunden
overlap=0.6; %?berlappung der frames
wlen = fs*frametime;
h = floor(wlen*(1-overlap));
nfft = wlen;
[stft, f, t] = stft(y, wlen, h, nfft, fs);



%Psychoakustische Redundanz entfernen
%S=size(stft);
%k=floor(wlen/20);%die Halbwertssamplel?nge (max. wlen/2)
%rstft=reduce(stft,k);
%Groesse=size(rstft)
%Reduktionsfaktor=length(rstft(:,1))/S(1)

%Schreibe Feature in csv-file
csvwrite('features.csv',transpose(abs(stft)));

%Rekonstruiere stft
%stft=reproduce(rstft,k,S(1));

%Spektrogramm erzeugen
imagesc(t, f, abs(stft));
axis xy

%Spektrogramminversion
[x_istft, t_istft] = istft(abs(stft), h, nfft, fs);

%Spiele spektrogramminvertiertes Signal ab
sound(x_istft,fs);

