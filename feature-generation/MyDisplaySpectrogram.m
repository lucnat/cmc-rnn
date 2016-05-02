
function MyDisplaySpectrogram(spec, fs, frameStep)
[fftSize,frameCount] = size(spec);
windowSize = floor(fftSize/2);
freqs = (0:windowSize)*fs/fftSize;%fs/fftSize=Frequenzunschärfe pro fftSample
times = (1:frameCount) * frameStep/fs;%frameStep/fs=Timestep pro frame
magSpec = spec(1:windowSize+1,:);
imagesc(times, freqs, magSpec);
axis xy



