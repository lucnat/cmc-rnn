require 'rnn'
require 'CSV.lua'

local X = CSV.read('bounce.csv')     -- training data

-- hyper-parameters 
local batchSize = 1
local rho = 20 -- sequence length
local hiddenSize = 40
local inputDimension = X:size(2)
local lr = 0.1
local maxIt = 1000
local flag=1

local rnn

print(string.format("InputDimesion %d ; InputSize = %d ", inputDimension, X:size(1)))

if flag==1 then
   -- load the model (importand to reinitialise the model or give a different model-name)
   rnn = torch.load('flag_rnn.net')
else


   -- build simple recurrent neural network
   local r = nn.Recurrent(
      hiddenSize,
      nn.Linear(inputDimension, hiddenSize),    -- processes input Tensors
      nn.Linear(hiddenSize, hiddenSize),        -- feedbacks the previous output Tensor
      nn.Tanh(),                             -- non-linear Module used to process the element-wise sum of the input and feedback module outputs
      rho                                       -- maximum amount of backpropagation steps to take back in time
   )

   rnn = nn.Sequential()
      :add(r)
      :add(nn.Linear(hiddenSize, inputDimension))

   -- wrap the non-recurrent module (Sequential) in Recursor.
   -- This makes it a recurrent module
   -- i.e. Recursor is an AbstractRecurrent instance
   rnn = nn.Recursor(rnn, rho)

   --print(rnn)
end

local criterion = nn.MSECriterion()

local p = 1 -- data pointer

-- training
local iteration = 1
while true do
   if(p+rho > X:size(1)) then
      p = 1
   end
   local inputs, targets = {}, {} 
   for step=1,rho do
      inputs[step] = X[p+step-1]
      targets[step] = X[p+step]
   end
   p = p+rho

   -- 2. forward sequence through rnn
   
   rnn:zeroGradParameters() 
   rnn:forget() -- forget all past time-steps
   local outputs, err = {}, 0
   for step=1,rho do
      outputs[step] = rnn:forward(inputs[step])
      err = err + criterion:forward(outputs[step], targets[step])
   end
   
   if iteration%100==0 then
      print(string.format("Iteration %d ; loss = %f ; datapointer %d ", iteration, err,p))
   end

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

-- save the model
torch.save('flag_rnn.net', rnn)


function sample(seed, N)
   local samples = torch.zeros(N, inputDimension)
   samples[1] = rnn:forward(seed)
   for i=2,N do
      samples[i] = rnn:forward(samples[i-1])
   end
   return samples
end

local seed = torch.rand(inputDimension) -- X[1]
--local seed = X[1]
local samples = sample(seed, 200)
-- print(samples)
CSV.write(samples)
