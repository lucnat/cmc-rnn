require 'rnn'
require 'CSV.lua'

-- command line stuff
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-state_file','epoch42los33.3333.net','The State file from which to sample')
cmd:option('-input_dimension',163,'the dimension of the vectors')
cmd:option('-length',100,'amount of samples')
cmd:text()
params = cmd:parse(arg)

--load the model
local rnn = torch.load(params.state_file)
local inputDimension = params.input_dimension
local length = params.length

local seed = torch.rand(inputDimension)
local next = seed;

function printCSV(v)
	local s = ''
	for k=1,v:size(1)-1 do
		s = s..v[k]..','
	end
	s = s..v[v:size(1)]
	print(s)
end

for i=1,params.length do
	next = rnn:forward(next)
	printCSV(next)
	before = next
end

