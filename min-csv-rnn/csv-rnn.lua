-- Minimal implementation of vanilla RNN trained on csv data by Luca Naterop

function readCSV(file)
	-- Reads a csv file with comma delimiter and returns resulting tensor
	local csvFile = io.open(file, 'r')
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

X = readCSV('Symbol.csv') 	-- training data
N = X:size(1) 				-- amount of frames
m = X:size(2)				-- amount of samples

print('Data has ' .. N .. ' frames and ' .. m .. ' samples')

-- hyperparameters
hidden_size 	= 130
seq_length 		= 25
learning_rate 	= 5e-2

-- model parameters
Wxh = 	torch.rand(hidden_size,m)*0.01
Whh = 	torch.rand(hidden_size,hidden_size)*0.01
Why = 	torch.rand(m,hidden_size)*0.01
bh = 	torch.zeros(hidden_size)
by = 	torch.zeros(m)

function lossFunction(inputs, targets, hprev)
	-- returns loss and gradients

	-- wichtig: hs[t] entspricht dem hs[t-1] im python code
	local xs 	= torch.zeros(seq_length, m)
	local hs 	= torch.zeros(seq_length+1, hidden_size)
	local ys 	= torch.zeros(seq_length, m)
	local diff 	= torch.zeros(seq_length, m)

	hs[1] = hprev:clone()
	local loss = 0
	-- forward propagation
	for t=1,seq_length do
		xs[t] 	= inputs[t]									-- fetch inputs
		hs[t+1] = torch.tanh(Wxh*xs[t] + Whh*hs[t] + bh) 	-- hidden layer activations
		ys[t] 	= Why*hs[t+1] + by							-- predictions
		diff[t] = targets[t] - ys[t]						-- deltas
		loss 	= loss + diff[t]:dot(diff[t])				-- squared error
	end

	--backward propagation
	local dWxh = 	torch.zeros(hidden_size,m)*0.01
	local dWhh = 	torch.zeros(hidden_size,hidden_size)*0.01
	local dWhy = 	torch.zeros(m,hidden_size)*0.01
	local dbh = 	torch.zeros(hidden_size)
	local dby = 	torch.zeros(m)
	local dhnext = 	torch.zeros(hidden_size) 

	for t = seq_length,1,-1 do
		dy 		= ys[t] - targets[t]
		dWhy 	= dWhy + torch.ger(dy,hs[t+1])
		dby 	= dby + dy
		dh 		= Why:t() * dy + dhnext 	-- backprop into h
		dhraw	= torch.cmul(dh, torch.pow(hs[t+1],2)*(-1)+1) -- backprop through tanh nonlinearity
		dbh		= dbh + dhraw
		dWxh 	= dWxh + torch.ger(dhraw,xs[t])
		dWhh 	= dWhh + torch.ger(dhraw,hs[t])
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

function sample(seed, h, amount)
	-- samples N vectors based on seed and hprev
	local samples = torch.zeros(amount+1,m)
	samples[1] = seed
	x = seed
	for i=2,amount+1 do
		h = torch.tanh(Wxh*x + Whh*h + bh)
		local y = Why*h + by
		samples[i] = y
		x = y
	end
	return samples
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

function adagrad(Parameter, dParameter, Memory)
	-- performs adagrad on these tensors. Changes Param and Memory
	Memory = Memory + torch.cmul(dParameter,dParameter)
	Parameter = Parameter - torch.cdiv(dParameter,torch.sqrt(Memory + 1e-8))*learning_rate
	return Parameter, Memory
end

n = 0 -- iteration count
p = 1 -- data pointer
cycles = 0

-- memory variables for adagrad
mWxh = 	torch.zeros(hidden_size,m)
mWhh = 	torch.zeros(hidden_size,hidden_size)
mWhy = 	torch.zeros(m,hidden_size)
mbh = 	torch.zeros(hidden_size)
mby = 	torch.zeros(m)

smooth_loss = -math.log(1.0/m)*seq_length
hprev = torch.zeros(hidden_size) 
loss = 100

while true do
	-- check if we are at the end of training data
	if p + seq_length + 1 >= N or n == 0 then
		p = 1
		hprev = torch.zeros(hidden_size)
		cycles = cycles + 1
	end

	inputs = X:sub(p, p+seq_length-1)
	targets = X:sub(p+1, p+seq_length)

	if n % 1 == 0 then
		print('iter: '.. n ..', loss: '.. smooth_loss ..', pointer '.. p ..', cycles '..cycles)
	end

	loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFunction(inputs, targets, hprev)
	smooth_loss = smooth_loss * 0.999 + loss * 0.001

  	Wxh, mWxh = adagrad(Wxh, dWxh, mWxh)
  	Whh, mWhh = adagrad(Whh, dWhh, mWhh)
  	Why, mWhy = adagrad(Why, dWhy, mWhy)
  	bh, mbh = adagrad(bh, dbh, mbh)
  	by, mby = adagrad(by, dby, mby)

	p = p + seq_length
	n = n + 1
	if cycles == 300 then
		sampleInput = X[p]
		break
	end
end

function writeCSV(M)
	local out = assert(io.open("./samples.csv", "w")) -- open a file for serialization
	splitter = ","
	for i=1,M:size(1) do
	    for j=1,M:size(2) do
	        out:write(M[i][j])
	        if j == M:size(2) then
	            out:write("\n")
	        else
	            out:write(splitter)
	        end
	    end
	end
	out:close()
end

local samples = sample(sampleInput, hprev, 400)

writeCSV(samples)
