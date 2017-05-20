import numpy as np
import numpy.matlib
import scipy.io.wavfile

def reproduce(X,k,n): 
	s = X.shape
	Y = X[0:k,:]
	for i in range(1, int(np.floor(s[0]/k))):
		for j in range(0,k):
			Y = np.concatenate((Y, np.matlib.repmat(X[k*i+j,:],np.power(2,i),1)))

	Rest = s[0]%k
	i = int(np.floor(s[0]/k))
	for j in range(0, Rest):
		Y = np.concatenate((Y, np.matlib.repmat(X[k*i+j,:],np.power(2,i),1)))
	Y = Y[0:n,:]
	return Y

def istft(stft, h, nfft, fs):
	nfft = int(nfft)
	coln = stft.shape[1]
	xlen = int(nfft + (coln-1)*h)
	x = np.zeros([1, xlen])
	win = np.hamming(nfft)
	win = np.reshape(win, [1,win.shape[0]])
	win = np.transpose(win)
	if nfft%2 == 1:
		for b in range(0, h*coln, h):
			X = stft[:, b/h]
			X = np.concatenate(( X , np.conj(np.flipud(X[1:]))))
			Win = np.matlib.repmat(win, 1, 1+b/h)
			xprim = np.real(np.fft.ifft(X, axis=0))
			x[:,(b):(b+nfft)] = x[:,(b):(b+nfft)] + np.transpose(np.multiply(xprim, win))
	else:
		for b in range(0, h*coln, h):
			print(b)
			X = stft[:, b/h]
			A = np.flipud(X[1:-1])
			X = np.concatenate(( X , np.conj(np.flipud(X[1:-1]))))
			Win = np.matlib.repmat(win, 1, 1+b/h)
			xprim = np.real(np.fft.ifft(X, axis=0))
			xprim = np.reshape(xprim, [len(xprim),1])
			x[:,b:(b+nfft)] = x[:,b:(b+nfft)] + np.transpose(np.multiply(xprim, win))

	W0 = np.sum(np.multiply(win, win))
	x = np.multiply(x, h/W0)

	actxlen = len(x)
	t = np.linspace(0,actxlen-1,actxlen)/fs
	return x,t

def inverseAFilter(x, fs): 
	sz = len(x)
	x = np.reshape(x, [len(x), 1])
	xlen = len(x)
	NumUniquePts = int(np.ceil((xlen+1)/2));
	X = np.fft.fft(x)
	X = X[0:NumUniquePts]		# FFT symmetric, throw away second half
	f = np.linspace(0,NumUniquePts-1,NumUniquePts)*fs/xlen
	c1 = 3.5041384e16
	c2 = 20.598997*20.598997
	c3 = 107.65265*107.65265
	c4 = 737.86223*737.86223
	c5 = 12194.217*12194.217
	shift = 1
	f = np.multiply(f+shift, f+shift)
	num = c1*np.power(f,4)
	den = np.multiply(np.power(c2+f,2), np.multiply(c3+f,np.multiply(c4+f,np.power(c5+f,2))))
	A = np.divide(num,den)
	A = np.reshape(A,[len(A),1])
	XA = np.divide(X,A)

	if np.fmod(xlen, 2) == 1:
		XA = np.concatenate(( XA , np.conj(np.flipud(XA[1:]))))
	else:
		XA = np.concatenate(( XA , np.conj(np.flipud(XA[1:-1]))))
	xA = np.real(np.fft.ifft(XA))
	return xA	

def writeAudioFromSamples(X):
	# Designparameter
	fs = 12000
	frametime = 0.08
	overlap = 0.6
	p = 2
	S = X.shape
	SIZE = [481, 163] 			# compression parameters
	wlen = fs*frametime
	k = int(np.floor(wlen/(p*10)))
	X = reproduce(X,k,SIZE[0])

	# koennte Spektrogramm plotten...

	h = int(np.floor(wlen*(1-overlap)))
	nfft = wlen
	x_istft, t_istft = istft(X,h,nfft,fs)
	x_istft = np.reshape(x_istft, [x_istft.shape[1]])
	filtered = inverseAFilter(x_istft, fs)
	filtered = np.reshape(filtered, [filtered.shape[0]])
	scipy.io.wavfile.write('mit.wav',fs,x_istft)

# # test
x = np.genfromtxt('test.csv', delimiter=',')
# x = inverseAFilter(x,44100)
scipy.io.wavfile.write('test.wav',44100,x)
