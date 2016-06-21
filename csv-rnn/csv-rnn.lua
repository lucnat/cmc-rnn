require 'rnn'
require 'CSV.lua'

X = CSV.read('cyclic.csv')     -- training data

-- hyper-parameters 
batchSize = 1
rho = 20 -- sequence length
hiddenSize = 40
inputDimension = X:size(2)
lr = 0.1
maxIt = 10000

print('input dimension = '..inputDimension..' input size = '..X:size(1))

-- build simple recurrent neural network
r = nn.Recurrent(
   hiddenSize,
   nn.Linear(inputDimension, hiddenSize),    -- processes input Tensors
   nn.Linear(hiddenSize, hiddenSize),        -- feedbacks the previous output Tensor
   nn.Sigmoid(),                             -- non-linear Module used to process the element-wise sum of the input and feedback module outputs
   rho                                       -- maximum amount of backpropagation steps to take back in time
)

rnn = nn.Sequential()
   :add(r)
   :add(nn.Linear(hiddenSize, inputDimension))

-- wrap the non-recurrent module (Sequential) in Recursor.
-- This makes it a recurrent module
-- i.e. Recursor is an AbstractRecurrent instance
rnn = nn.Recursor(rnn, rho)

print(rnn)

criterion = nn.MSECriterion()

p = 1 -- data pointer

-- training
iteration = 1
while true do
   if(p+rho > X:size(1)) then
      p = 1
   end
   inputs, targets = {}, {} 
   for step=1,rho do
      inputs[step] = X[p+step-1]
      targets[step] = X[p+step]
   end
   p = p+rho

   -- 2. forward sequence through rnn
   
   rnn:zeroGradParameters() 
   rnn:forget() -- forget all past time-steps
   outputs, err = {}, 0
   for step=1,rho do
      outputs[step] = rnn:forward(inputs[step])
      err = err + criterion:forward(outputs[step], targets[step])
   end

   print('Iteration = '..iteration..' loss = '..err)

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   local gradOutputs, gradInputs = {}, {}
   for step=rho,1,-1 do -- reverse order of forward calls
      gradOutputs[step] = criterion:backward(outputs[step], targets[step])
      gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
   end

   -- 4. update
   
   rnn:updateParameters(lr)
   
   iteration = iteration + 1

   if( iteration >= maxIt) then
      break
   end

end

function sample(seed, N)
   local samples = torch.zeros(N, inputDimension)
   samples[1] = rnn:forward(seed)
   for i=2,N do
      samples[i] = rnn:forward(samples[i-1])
   end
   return samples
end

local seed = torch.rand(inputDimension) -- X[1]
local samples = sample(seed, 300)
-- print(samples)
CSV.write(samples)
