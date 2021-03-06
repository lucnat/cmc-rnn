%Feature to Audio

clear, clc, close all

%take same parameter as in the feature transformation!
fs=8000;
frameStep = 64;
n=256;%=original windowsize
k=32;

iterations = 40; %program is universal w.r.t. iteration
filename='generated.wav';

%read in feature
disp('reading csv...');
magspec_red=csvread('sound.csv')';

%reconstruct the frequency resolution redundance magnitude spectrum
disp('reconstruction...');
magspec=reproduce(magspec_red,k,n);

%load the redundant negativ frequency amplitudes
magspec_complete=[magspec;flipud(magspec)]; 

%invert the complete amplitude spectrum
[y_reconstructed, ~] = InvertSpectrogram(magspec_complete, frameStep,iterations);
y_reconstructed=y_reconstructed(y_reconstructed<1e10);%remove NaNs

%inverted A-filter
disp('applying inverse A filter...');
y_reconstructed=inverse_filterA(y_reconstructed,fs);

%sound the reconstructed signal
%sound(y_reconstructed,fs)

%store the reconstructed signal
disp('writing audio file...');
audiowrite(filename,y_reconstructed,fs)
disp('done');
