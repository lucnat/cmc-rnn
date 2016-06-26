require 'rnn'
require 'CSV.lua'

--load the model
local rnn = torch.load('flag_rnn.net')
local X = CSV.read('features_short.csv')   -- training data

inputDimension = X:size(2)
inputSize = X:size(1)

function sample(seed, N)
   local samples = torch.zeros(N, inputDimension)
   samples[1] = rnn:forward(seed)
   for i=2,N do
      samples[i] = rnn:forward(samples[i-1])
   end
   return samples
end

for i=1,math.floor(inputSize/2) do
	rnn:forward(X[i])
end

local s='samples'
for i=1,10 do
	--local seed = torch.rand(inputDimension)
	for j=1,math.floor(inputSize/(11-i)-1) do
		rnn:forward(X[i])
    end
	local seed = X[math.floor(inputSize/(11-i))] 
	local samples = sample(seed, 300)
	CSV.write(samples,s..tostring(i))
end