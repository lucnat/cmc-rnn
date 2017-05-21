function Xb= reduce(X,k)
    %X, das auszudünnende Spektrogramm (STFFT als Spaltenvektoren)
    %k, die Binlängen in #Samples, in welche die Frequenzen zusammen gefasst werden.
    %bin1: k->k
    %bin2: 2k->k
    %bin3: 3k->k
    %...
    
    s=size(X);
    l=s(1);
    
    if l>k
        %In den bins wird die Frequenzauflösung spezifisch ausgedünnt
        bin=[1;k];
        i=2;
        while k*2^(i-1)+bin(i)<l,
            bin=[bin;k*2^(i-1)+bin(i)];
            i=i+1;
        end
              
        %Berechne durchschnittliche Frequenzen in den bins
        %bin1-2: direkt übernehmen
        %bin2-3: 2^1-er Paare
        %bin3-4: 2^2_er Gruppen
        %...
        Xb=[X(bin(1):bin(2),:)];
        for i=2:length(bin)-1
            Xbb=X(bin(i)+1:2^(i-1):bin(i+1),:);
            for j=2:2^(i-1)%jmax=2,4,8,...
                Xbb=Xbb+X(bin(i)+j:2^(i-1):bin(i+1),:); 
            end
            Xb=[Xb; Xbb/2^(i-1)];
        end
        L=length(bin);
        Restloops=floor((l-bin(end))/2^(L-1));
        for i=0:Restloops-1
            Xb=[Xb;mean(X(bin(end)+1+i*2^(L-1):bin(end)+(i+1)*2^(L-1),:),1)];
        end
        if mod(l-bin(end),2^(length(bin)-1))>0
            Xb=[Xb;mean(X(end+1-mod(l-bin(end),2^(L-1)):end,:),1)];
        end
    else
    Xb=X;
    end
    
end
