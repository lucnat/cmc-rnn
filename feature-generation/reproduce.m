function Y = reproduce(X,k,n)
    s=size(X);
    Y=X(1:k,:);
    for i=1:floor(s(1)/k)-1
        for j=1:k
            Y=[Y;repmat(X(k*i+j,:),2^i,1)];
        end
    end
    Rest=mod(s(1),k);
    i=floor(s(1)/k);
    for j=1:Rest
        Y=[Y;repmat(X(k*i+j,:),2^i,1)];
    end
    Y=Y(1:n,:);
    
end

