-- Minimal implementation of vanilla RNN trained on csv data by Luca Naterop

function readCSV(file)
	-- Reads a csv file with comma delimiter and returns resulting tensor
	local csvFile = io.open('cyclic.csv', 'r')
	local firstLine = csvFile:read():split(',')
	local m = table.getn(firstLine)
	local N = csvFile:seek("end")/(2*m)
	csvFile:seek("set")        -- restore position

	local X = torch.zeros(N,m)
	local i = 0
	for line in csvFile:lines('*l') do
		i = i + 1
		local l = line:split(',')
		for key,val in ipairs(l) do
			X[i][key] = val
		end
	end
	return X
end

X = readCSV('cyclic.csv') 	-- training data
N = X:size(1) 				-- amount of frames
m = X:size(2)				-- amount of samples

print('Data has ' .. N .. ' frames and ' .. m .. ' samples')

-- hyperparameters
hidden_size 	= 100
seq_length 		= 25
learning_rate 	= 1e-1

-- model parameters
Wxh = 	torch.rand(hidden_size,m)*0.01
Whh = 	torch.rand(hidden_size,hidden_size)*0.01
Why = 	torch.rand(m,hidden_size)*0.01
bh = 	torch.rand(hidden_size)
by = 	torch.rand(m)

function lossFunction(inputs, targets, hprev)
	-- returns loss and gradients

	-- wichtig: hs[t] entspricht dem hs[t-1] im python code
	local xs 	= torch.zeros(seq_length+1, m)
	local hs 	= torch.zeros(seq_length+1, hidden_size)
	local ys 	= torch.zeros(seq_length+1, m)
	local diff 	= torch.zeros(seq_length+1, m)

	hs[1] = hprev:clone()
	local loss = 0

	-- forward propagation
	for t=2,seq_length+1 do
		xs[t] 	= inputs[t-1]								-- fetch inputs
		hs[t] 	= torch.tanh(Wxh*xs[t] + Whh*hs[t-1] + bh) 	-- hidden layer activations
		ys[t] 	= Why*hs[t] + by							-- predictions
		diff[t] = targets[t-1] - ys[t]						-- deltas
		loss 	= loss + diff[t]:dot(diff[t])				-- squared error
	end

	--backward propagation
	local dWxh = 	torch.zeros(hidden_size,m)*0.01
	local dWhh = 	torch.zeros(hidden_size,hidden_size)*0.01
	local dWhy = 	torch.zeros(m,hidden_size)*0.01
	local dbh = 	torch.zeros(hidden_size)
	local dby = 	torch.zeros(m)
	local dhnext = 	torch.zeros(hidden_size) 

	for t = seq_length+1,2,-1 do
		dy 		= ys[t] - targets[t-1]
		dWhy 	= dWhy + torch.ger(dy,hs[t])
		dby 	= dby + dy
		dh 		= Why:t() * dy + dhnext 	-- backprop into h
		dhraw 	= dh * (1 - hs[t] * hs[t])  -- backprop through tanh nonlinearity
		dbh		= dbh + dhraw
		dWxh 	= dWxh + torch.ger(dhraw,xs[t])
		dWhh 	= dWhh + torch.ger(dhraw,hs[t-1])
		dhnext 	= Whh:t()*dhraw
	end

	-- clip to mitigate exploding gradients
	dWxh:apply(clip)
	dWhh:apply(clip)
	dWhy:apply(clip)
	dbh:apply(clip)
	dby:apply(clip)

	return loss, dWxh, dWhh, dWhy, dbh, dby, hs[seq_length+1]
end

function sample(seed, hprev, N)
	-- samples N vectors based on seed and hprev

	return
end

function clip(element)
	-- clips element to stay between [-5,5]

	if element > 5 then 
		return 5
	elseif element < -5 then 
		return -5
	else 
		return element
	end
end


n = 0 -- iteration count
p = 1 -- data pointer

-- memory variables for adagrad
mWxh = 	torch.zeros(hidden_size,m)
mWhh = 	torch.zeros(hidden_size,hidden_size)
mWhy = 	torch.zeros(m,hidden_size)
mbh = 	torch.zeros(hidden_size)
mby = 	torch.zeros(m)

smooth_loss = -math.log(1.0/m)*seq_length
hprev = torch.zeros(hidden_size) 

 while true do

	-- check if we are at the end of training data
	if p + seq_length + 1 >= N or n == 0 then
		p = 1
	end

	inputs = X:sub(p, p+seq_length-1)
	targets = X:sub(p+1, p+seq_length)

	-- print('inputs')
	-- print(inputs)
	-- print('targets')
	-- print(targets)

	loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFunction(inputs, targets, hprev)
	smooth_loss = smooth_loss * 0.999 + loss * 0.001

	if n % 100 == 0 then
		print('iter: '.. n ..', loss: '.. smooth_loss ..', pointer: '.. p) -- print progress
		-- sample(hprev,features[p],5)
	end

	p = p + seq_length
	n = n + 1

 end








