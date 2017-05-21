function DisplaySpectrogram(spec, fs, frameStep, enhanceHighFrequencies)
% function DisplaySpectrogram(spec, fs, frameStep, enhanceHighFrequencies)
% Display a (potentially) complex-valued, positive-and-negative-frequency
% spectrogram in the normal format.
% spec - a spectrogram with positive and negative frequencies.  Each column
% contains the spectral information (positive and negative frequencies) for
% one frame's worth of data.
% fs - Sample rate for the original waveform data
% frameStep - The number of samples between spectrogram frames.
% enhanceHighFrequencies - Modify the the spectrogram to enhance high
% frequencies.  This is often useful because speech has a 1/f spectrum, and
% by multiplying each spectral bin by its frequency we can counteract this.
% High-frequency enhancement is the default when using DisplaySpectrogram.

% Written by Malcolm Slaney, Microsoft Research, July 2014
if nargin < 4
    enhanceHighFrequencies = 1;
end

[fftSize,frameCount] = size(spec);
windowSize = floor(fftSize/2);

freqs = (0:windowSize)*fs/fftSize;
times = (1:frameCount) * frameStep/fs;

posFreqs = spec(1:windowSize+1, :);
negFreqs = flipud(spec(windowSize+1:end,:));

powerSpec = posFreqs;
freqRange = 2:windowSize+1;
% Compute the power.  Easy since positive and negative values are
% conjugates.
powerSpec(freqRange,:) = powerSpec(freqRange,:).*negFreqs;

if enhanceHighFrequencies
    magSpec = sqrt(powerSpec) .* repmat((0:length(freqs)-1)', 1, frameCount);
else
    magSpec = sqrt(powerSpec);
end
if 1
    imagesc(times, freqs, real(magSpec));
    axis xy
else                            % Just for debugging
    plot(freqs, 20*log10(real(magSpec)));
end