function xA = inverse_filterA(x, fs)

% function: xA = filterA(x, fs)
% x - signal in the time domain
% fs - sampling frequency, Hz
% xA - filtered signal in the time domain

% determine the signal size
sz = size(x);

% represent x as column-vector
x = x(:);

% signal length
xlen = length(x);

% number of unique points
NumUniquePts = ceil((xlen+1)/2);

% FFT
X = fft(x);

% fft is symmetric, throw away second half
X = X(1:NumUniquePts);


% frequency vector with NumUniquePts points
f = (0:NumUniquePts-1)*fs/xlen;

%plot(f,abs(X))

% A-weighting filter coefficients
c1 = 3.5041384e16;
c2 = 20.598997^2;
c3 = 107.65265^2;
c4 = 737.86223^2;
c5 = 12194.217^2;

% evaluate the A-weighting filter in the frequency domain
%for invertibility for positive f
f_low=f(f<=100);
f_high=f(f>100);

fq = f_high.^2;
num = c1*fq.^4;
den = ((c2+fq).^2) .* (c3+fq) .* (c4+fq) .* ((c5+fq).^2);
A = num./den;
A = A(:);
A = [A(1)*ones(length(f_low),1);A];%make it a continous function


%plot(f,1./A)

% filtering in the frequency domain
XA = X./A;



% reconstruct the whole spectrum
if rem(xlen, 2)                     % odd xlen excludes the Nyquist point
    XA = [XA; conj(XA(end:-1:2))];
else                                % even xlen includes the Nyquist point
    XA = [XA; conj(XA(end-1:-1:2))];
end

% IFFT
xA = real(ifft(XA));
    
% represent the filtered signal in the form of the original one
xA = reshape(xA, sz);

%normalize
xA = xA/max(xA);

end

