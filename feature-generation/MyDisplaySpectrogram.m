
function MyDisplaySpectrogram(spec, fs, frameStep)
% function DisplaySpectrogram(spec, fs, frameStep, enhanceHighFrequencies)
% Display a (potentially) complex-valued, positive-and-negative-frequency
% spectrogram in the normal format.
% spec - a spectrogram with ONLY positive frequencies.  Each column
% contains the spectral information for
% one frame's worth of data.
% fs - Sample rate for the original waveform data
% frameStep - The number of samples between spectrogram frames.
% enhanceHighFrequencies - Modify the the spectrogram to enhance high
% frequencies.  This is often useful because speech has a 1/f spectrum, and
% by multiplying each spectral bin by its frequency we can counteract this.
% High-frequency enhancement is the default when using DisplaySpectrogram.

% Written by Malcolm Slaney, Microsoft Research, July 2014


Fs=fs;
fs=floor(10000); %Samplerate für Frequenzen bis fs/2 Hz
spec=resample(spec,fs,Fs); %Interpoliert/dezimiert das Signal für fs

[fftSize,frameCount] = size(spec);
windowSize = floor(fftSize/2);

freqs = (0:windowSize)*fs/fftSize;%fs/fftSize=Anzahl Sequenzen pro Sekunde
times = (1:frameCount) * frameStep/fs;



% Compute the power.  Easy since positive and negative values are

powerSpec = abs(spec(1:windowSize+1,:));

%multipliziert jede Frequenzenergie mit der zugehörigen Frequenz 
%(frequenz - samplepunkte entsprechung)

    magSpec = sqrt(powerSpec);


    imagesc(times, freqs, real(magSpec'));%magSpec sollte doch sowieso real sein..??
    axis xy



