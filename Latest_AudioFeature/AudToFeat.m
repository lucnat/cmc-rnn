%Audio to Feature

clear, clc, close all

windowSize = 256;%program is not universal to this parameter
frameStep = 64;%%program is not universal to this parameter
fs=8000;%program is universal w.r.t. fs, gets significantly wors over fs=12000

featurename='chopin_short_features.csv';

% read in audio file and resample
[y, Fs] = audioread('chopin_short.wav', [1200000 1500000]);   
y=sum(y,2);
%sound(y,Fs)
y=resample(y,fs,Fs);

%A-filtering
y=filterA(y,fs);

%generate complex spectrogram
compSpec = ComplexSpectrogram(y, windowSize, frameStep);

%plot magnitude spectrogram    
%clf
%DisplaySpectrogram(compSpec, fs, frameStep);

%generate magnitude spectrogram
magspec=abs(compSpec);
magspec=magspec(1:end/2,:);

%reduce the psycho acoustic redundance in high frequency resolution
k=32;
magspec_red=reduce(magspec,k);

%store the features
csvwrite(featurename,magspec_red);