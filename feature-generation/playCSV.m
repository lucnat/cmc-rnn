f='samples.csv';
FeatureMatrix=csvread(strcat('/Users/Giaco/Desktop/cmc-rnn/csv-rnn/',f));
size(FeatureMatrix)

%zeige smaples_Data
figure(1)
imagesc(FeatureMatrix)

%Spektrogramminversion
fs=3000;
frametime=0.1; %zeitliche Auflösung in Sekunden
overlap=0.6; %Überlappung der frames
wlen = fs*frametime;
h = floor(wlen*(1-overlap));
nfft = wlen;
[x_istft, t_istft] = istft(FeatureMatrix', h, nfft, fs);
%x_istft=InvPowSpec(stft,fs,h);

%Spiele spektrogramminvertiertes Signal ab
sound(x_istft,fs);