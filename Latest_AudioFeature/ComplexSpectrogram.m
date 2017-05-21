function spectrum = ComplexSpectrogram(waveform, windowSize, frameStep)
% function spectrum = ComplexSpectrogram(waveform, windowSize, frameStep)
% Compute the complex spectrogram of a one-dimensional waveform.  This
% returns complex values, for both positive (first half the of the rows)
% and negative (bottom half of the rows) frequencies of the spectrogram.
%
% The input waveform is the first argument and is one dimensional.
% windowSize is the amount of data (in samples) used for each column of the
% spectrogram.  This determines the frequency resolution of the resulting
% spectrogram. This resolution is 1 cycle per windowSize. Finally,
% frameStep is the number of samples between rows of the spectrogram.

% Unlike many spectrograms (of speech), this spectrogram does not enhance the
% high-frequency content of the signal (so that high formants are more
% visible.) This can be fixed in DisplaySpectrogram.

% Written by Malcolm Slaney, Microsoft Research, July 2014

if nargin < 2
    windowSize = 256;
end
if nargin < 3
    frameStep = 128;
end

% Figure out the fftSize (twice the window size because we are doing
% circular convolution).  We'll place the windowed time-domain signal into
% the middle of the buffer (zeros before and after the signal in the array.)
fftSize = 2*windowSize;
fftB = floor(windowSize/2);
fftE = fftB + windowSize - 1;
fftBuffer = zeros(1,fftSize);

[r,c] = size(waveform);
if r > c
    waveform = waveform';
end

frameCount = floor((length(waveform) - windowSize)/frameStep) + 1;

spectrum = zeros(fftSize, frameCount);
% h = hamming(windowSize)';
h = 0.54 - .46*cos(2*pi*(0:windowSize-1)/(windowSize-1));
% h = h * 0 + 1;              % Just for debugging, no window.

% Note: This code loads the waveform data (times hamming) into the center
% of the fftSize buffer.  Then uses fftshift to rearrange things so that
% the 0-time is Matlab sample 1.  This means that the center of the window
% defines 0 phase.  After ifft, zero time will be at the same place.
for frameNumber = 1:frameCount
     waveB = 1 + (frameNumber-1)*frameStep;
     waveE = waveB + windowSize - 1;
     fftBuffer = 0*fftBuffer;           % Make sure the entire buffer is empty
     fftBuffer(fftB:fftE) = waveform(waveB:waveE) .* h;
     fftBuffer = fftshift(fftBuffer);
     % fftBuffer(fftE+1:end) = 0;
     % transpose (without the conjugate) into a column vector.
     spectrum(:,frameNumber) = fft(fftBuffer).';
end

function w=MyHamming(N)


