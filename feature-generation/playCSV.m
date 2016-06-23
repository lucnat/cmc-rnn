f='samples.csv';
FeatureMatrix=csvread('samples.csv',f);
size(FeatureMatrix)

%zeige smaples_Data
figure(1)
imagesc(FeatureMatrix)

%Spektrogramminversion
fs=20000;
frametime=0.1; %zeitliche Auflösung in Sekunden
overlap=0.8; %Überlappung der frames


wlen = fs*frametime;
h = floor(wlen*(1-overlap));
nfft = wlen;
[x_istft, t_istft] = istft(FeatureMatrix', h, nfft, fs);
%x_istft=InvPowSpec(stft,fs,h);

%Spiele spektrogramminvertiertes Signal ab
sound(x_istft,fs);
