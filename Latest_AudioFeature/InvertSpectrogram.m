function [waveform, err] = InvertSpectrogram(originalSpectrogram, frameStep, ...
    iterationCount, debugPlots, estimateTimeDelay, randomizePhase)
% function [waveform, err] = InvertSpectrogram(originalSpectrogram, frameStep, ...
%     iterationCount, debugPlots, estimateTimeDelay, randomizePhase)
% Find the best waveform that approximates the input spectrogram. This is
% known as spectrogram inversion.
% 
% The originalSpectrogram must be a full spectrogram (usually magnitude
% only) containing both the positive and negative frequencies in normal FFT
% order (freqs: 0 1 2 3 4 ... +/-N/2 -N/2-1... -4 -3 -2 -1).
% frameStep is the number of samples between time-domain windows.
% iterationCount is the desired number of iterations, and usually 10 is
% enough.
%
% debugPlots shows several plots useful for debugging.  There is a pause
% between iterations, so be sure to hit a carriage return to advance to the
% next plot.
%
% estimateTimeDelay is true (non-zero) by default. Turning this flag off is
% useful to test and compare the performance with this important efficiency
% improvement.
% 
% randomizePhase is false (zero) by default.  Turning this flag on is
% useful to test and compare performance when starting with random phase
% instead of zero.

% The basic algorithm for this code is based on 
% D. Griffin and J. Lim. Signal estimation from modified short-time
%     Fourier transform. IEEE Trans. Acoust. Speech Signal Process.,
%     32(2):236-243, 1984.
% An important efficiency improvement is based on estimating a most
% harmonious phase when doing the first iteration (Griffin and Lim assumes
% zero phase).  This idea was first used in 
%     Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory Model
%     Inversion for Sound Separation, Proc. IEEE-ICASSP, Adelaide,
%     1994, II.77-80.
% and formally described in 
%     Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal Estimation From 
%     Modified Short-Time Fourier Transform Magnitude Spectra. 
%     IEEE Transactions on Audio Speech and Language Processing, 08/2007; 

% Written by Malcolm Slaney, Microsoft Research, July 2014
% Sept. 18, 2014 - Added real after ifft.   Slight improvement in result
%   for case with time-estimate and random phase (because time estimate was
%   confused with complex data.)


if nargin < 4
    debugPlots = 0;
end
if nargin < 5
    estimateTimeDelay = 1;
end
if nargin < 6
    randomizePhase = 0;
end

[fftSize, ~] = size(originalSpectrogram);
windowSize = floor(fftSize / 2);

currentSpectrogram = originalSpectrogram;
magOrigSpectrogram = abs(originalSpectrogram); % In case we got a complex spectrogram
regularization = max(max(magOrigSpectrogram)) / 1e8;

for iteration = 1:iterationCount
    % Invert the spectrogram by summing and adding
    waveform = InvertOneSpectrogram(currentSpectrogram, frameStep, ...
        iteration, debugPlots, estimateTimeDelay, randomizePhase);
    if debugPlots
        figure(1);
        subplot(3,1,1);         % Plot the whole waveform
        plot(waveform); 
        title(sprintf('Waveform Summation, at start of iteration %d', iteration));
        xlim([1 length(waveform)]);
        subplot(3,1,2);        % Plot the first transition
        plot(waveform);
        hold on; plot([frameStep frameStep], [-1 1], 'r--'); hold off
        xlim([frameStep-50 frameStep+50]);
        subplot(3,1,3);         % Plot a later transition
        plot(waveform);
        hold on; plot([4*frameStep 4*frameStep], [-1 1], 'r--'); hold off
        xlim([4*frameStep-50 4*frameStep+50]);
    end
    
    % Compute the resulting (complex) spectrogram
    newSpectrogram = ComplexSpectrogram(waveform, windowSize, frameStep);
    if nargout > 1
        diff = abs(newSpectrogram) - magOrigSpectrogram;
        err(iteration) = sum(sum(diff.^2));
    end
    if 0 && debugPlots
        figure(3); clf;
        if 1
            frameNumber = 3;
            plot(1:size(magOrigSpectrogram, 1), magOrigSpectrogram(:,frameNumber), ...
                1:size(newSpectrogram, 1), abs(newSpectrogram(:,frameNumber)), 'r--');
            legend('Original', 'New Estimate');
            title(sprintf('Spectral Slice %d of Spectrograms', frameNumber));
        else
            colormap(1-gray);
            clf; subplot(3,1,1);
            imagesc(magOrigSpectrogram); title('Original'); colorbar
            subplot(3,1,2);
            imagesc(abs(newSpectrogram)); title('New Estimate'); colorbar
            subplot(3,1,3);
            diff = magOrigSpectrogram - abs(newSpectrogram);
            imagesc(diff); title('Difference'); colorbar;
        end
    end
    if debugPlots > 0
        pause;
    end
    % Keep the original magnitude, but use the new phase (make sure we
    % don't divide by zero.)
    newPhase = newSpectrogram ./ max(regularization, abs(newSpectrogram));
    currentSpectrogram = magOrigSpectrogram .* newPhase;
end
waveform = InvertOneSpectrogram(currentSpectrogram, frameStep, ...
    iterationCount+1, debugPlots, estimateTimeDelay);



function waveform = InvertOneSpectrogram(originalSpectrogram, frameStep, ...
    iterationNumber, debugPlots, estimateTimeDelay, randomizePhase)
[fftSize, frameCount] = size(originalSpectrogram);
windowSize = floor(fftSize/2);

waveform = zeros(1, frameCount * frameStep + windowSize - 1);
totalWindowingSum = waveform;                 % To sum up total windowing effect
% h = hamming(windowSize)';
h = 0.54 - .46*cos(2*pi*(0:windowSize-1)/(windowSize-1));

fftB = floor(windowSize/2);
fftE = fftB + windowSize - 1;

for frameNumber=1:frameCount
    waveB = 1 + (frameNumber-1)*frameStep;
    waveE = waveB + windowSize - 1;
    if isreal(originalSpectrogram)
        spectralSlice = complex(originalSpectrogram(:,frameNumber));
        if randomizePhase && iterationNumber == 1
            % randomize the phase
            spectralSlice = spectralSlice.*exp(1i*2*pi*rand(size(spectralSlice)));
        end
    else
        spectralSlice = originalSpectrogram(:,frameNumber); % Already complex
    end
    newFrame = ifft(spectralSlice).';
    newFrame = real(fftshift(newFrame));            % Added Sept. 18, 2014
    if estimateTimeDelay > 0 && iterationNumber == 1 && frameNumber > 1
        offsetSize = windowSize-frameStep;
        bestOffset = FindBestOffset(waveform(waveB:(waveB+offsetSize-1)), ...
            newFrame(fftB:(fftB+offsetSize-1)));
    else
        bestOffset = 0;
    end
    if debugPlots && iterationNumber == 1
        figure(3); clf;
        unshiftedPortion = [zeros(1,waveB-1) newFrame(fftB:fftE)];
        shiftedPortion = [zeros(1,waveB-1) newFrame(-bestOffset + (fftB:fftE))];
        plot(1:length(waveform), waveform, 'b', ...
            1:length(unshiftedPortion), unshiftedPortion, 'r--', ...
            1:length(shiftedPortion), shiftedPortion, 'k-.');
        xlim([max(1,waveB-fftB) waveE]);
        legend('Current sum', 'Unshifted section to add', 'Shifted section to addd');
        title(sprintf('Waveform from InvertOneSpectrogram: Frame %d', frameNumber));
        pause
    end
    if 0
        plot(1:(waveE-waveB+1), waveform(waveB:waveE), ...
            1:(fftE-fftB+1), newFrame(-bestOffset + (fftB:fftE)))
    end
    waveform(waveB:waveE) = waveform(waveB:waveE) + ...
        newFrame(-bestOffset + (fftB:fftE));
    totalWindowingSum(waveB:waveE) = totalWindowingSum(waveB:waveE) + h;
end
waveform = real(waveform) ./ totalWindowingSum;

function bestOffset = FindBestOffset(waveform, newFrame)
frameSize = length(newFrame);
frameHalfSize = floor(frameSize/2);

cor = fastXcorr(waveform, newFrame);
% Remove portions that are outside the normal window area. Might want to be
% more conservative.
cor(1:frameHalfSize) = nan;
cor(floor(length(cor)-frameHalfSize):end) = nan;

[m,i] = max(cor);
bestOffset = i-(length(waveform)+1);
if bestOffset < -frameHalfSize || bestOffset > frameHalfSize
   error('Bad best offset found.');
end

    
if 0
    figure(3);
    subplot(3,1,1);
    plot(1:length(waveform), waveform, (1:length(newFrame)), newFrame);
    title('FindBestOffset testing, before alignment');
    legend('Current OLA waveform', 'Next portion');
    
    % subplot(3,1,2);
    % plot(1:length(waveform), waveform, 1:length(waveform), newFrame(windowSize/2:(windowSize/2+windowSize-1)))
    
    subplot(3,1,2);
    plot(cor);
    hold on; plot(i,m,'rx'); hold off;
    title('Correlation between existing and new frames');
    
    subplot(3,1,3);
    plot(1:length(waveform), waveform, (1:length(newFrame))+bestOffset, newFrame);
    title('FindBestOffset testing, after alignment');
    legend('Current OLA waveform', 'Next portion');
    
end


function y = fastXcorr(x1, x2)
% y = fastXcorr(x1, x2)
% Compute the cross correlation using the FFT. 

% Take advantage of the fact that Matlab's FFT works for any length vector
X1 = fft([x1 0*x1]);
X2 = fft([x2 0*x2]);

y = fftshift(ifft(X1 .* conj(X2)));

y = y(2:end);
