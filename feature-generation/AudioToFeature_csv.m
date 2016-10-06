clear, clc, close all

%Designparameter:
fs=12000; 
frametime=0.08; %zeitliche Aufl?sung in Sekunden
overlap=0.6; %?berlappung der frames
p=2; %psychoakustische redundanz (je grösser desto mehr reduktion)
red=0.8; %pvd reduktion (je kleiner desto mehr reduktion)

%dimensionalityreduction (setze 1 für Aktivierung)
psychoakustik=1;
SVD=0;

%einlesen und resampeln 6h_classic_piano
[y1,Fs1] = audioread('Bach2.mp3',[1,100000]);
[y2,Fs2] = audioread('Bach3.mp3',[1,100000]);
Fs1-Fs2
y1=sum(y1,2);
y2=sum(y2,2);
y=resample([y1;y2],fs,Fs1);


%STFT erzeugen
wlen = fs*frametime;
h = floor(wlen*(1-overlap));
nfft = wlen;
[stft, f, t] = stft(y, wlen, h, nfft, fs);
X=abs(stft); %kicke phase weg
featuredimensions=size(X)
% figure(1)
% imagesc(X)

%------------------------------------------------------

%Psychoakustische Redundanz entfernen
psychoreduction=1;
S=size(X);
if(psychoakustik==1)
    k=floor(wlen/(p*10));%die Halbwertssamplel?nge (max. wlen/2)
    X=reduce(X,k);
    Sr=size(X);
    psychoreduction=length(X(:,1))/S(1)
end


%SVD
svdreduction=1;
if(SVD==1)
    Xs=size(X);
    s=svd(X');
    [U,S,V] = svd(X');
    crit_sum=red*sum(s);
    bin=0;
    i=0;
    while bin<crit_sum
        i=i+1;
        bin=bin+s(i);
    end

    Sr=S(:,1:i);% keep the large singular values only
    X=(U*Sr)'; %the dim'reduced feature Matrix
    Rs=size(X);
    Vr=V(:,1:i)';
    svdreduction=Rs(1)/Xs(1)  
    csvwrite('svd_backtrafo.csv',Vr);
end

Sr=size(X)
reducedfeaturedimension=Sr(1)
Totalereduktion=svdreduction*psychoreduction

% figure(2)
% imagesc(X)

%Schreibe csv-feature
csvwrite('features.csv',X);
csvwrite('size.csv',[S(1) Sr(1)]);



%------








