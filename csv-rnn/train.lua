require 'rnn'
require 'CSV.lua'
require 'optim'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-csv_file','bounce.csv','the csv file that you would like to train')
cmd:text()

params = cmd:parse(arg)

print("Reading Data...")
local X = CSV.read(params.csv_file)     -- training data
print("Done.")

-- hyper-parameters 
batchSize       = 1
rho             = 50 -- sequence length
hiddenSize      = 1024
inputDimension  = X:size(2)
lr              = 0.05
maxIt           = 100000
loadFile        = 0

print("InputDimension = "..inputDimension.."; InputSize = ".. X:size(1))

if loadFile==1 then
   -- load the model (importand to reinitialise the model or give a different model-name)
   rnn = torch.load('CHANGE_MY_NAME.net')
else
   -- build simple recurrent neural network
   rnn = nn.Sequential()
      :add(nn.LSTM(inputDimension,hiddenSize,rho))
      :add(nn.LSTM(hiddenSize,hiddenSize,rho))
      :add(nn.Linear(hiddenSize, inputDimension))

   -- wrap the non-recurrent module (Sequential) in Recursor.
   -- This makes it a recurrent module
   -- i.e. Recursor is an AbstractRecurrent instance
   rnn = nn.Recursor(rnn, rho)
   print(rnn)
end

 local params, grads = rnn:getParameters()
 criterion = nn.MSECriterion()
--criterion.sizeAverage = false --throws the normalisation (1/n) away (faster)

 local  p = 1 -- data pointer
 local smothloss=100
 local epoch=1


-- function für optim
local feval = function(x)
  if x ~= params then
    params:copy(x)
  end
  grads:zero()
  local loss=0
  local inputs, targets, outputs = {}, {}, {}
  
  -- forward
   for step=1,rho do
      inputs[step] = X[p+step-1]
      targets[step] = X[p+step]
      
      --ouput normal:
      --outputs[step] = rnn:forward(inputs[step])
      
      --Idee negative Amplitude-->0: (seems to converge faster!) 
        out = rnn:forward(inputs[step])
        idx=torch.lt(out, 0) --überall eins, falls kleiner als null, sonst null
        out[idx]=0 --überall wo idx==1 wird out zu null, sonst unverändert
        outputs[step]=out

      loss = loss + criterion:forward(outputs[step],targets[step])
   end
  
  -- backward
  local gradOutputs, gradInputs = {}, {}
  for step=rho,1,-1 do -- reverse order of forward calls
     gradOutputs[step] = criterion:backward(outputs[step], targets[step])
     gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])--unnecessary save of gradInputs
  end
  return loss, grads
end

------------------------------------------------------------------------
-- optimization loop (with optim)
local optim_state = {lr}
for i = 1, maxIt do
   if(p+rho > X:size(1)) then
        p = 1
        epoch=epoch+1
   end
   
  local _, loss = optim.adagrad(feval, params, optim_state)
  smothloss=0.95*smothloss+0.05*loss[1]

  if i % 1 == 0 then
    print("Iteration = ".. i ..", Smothloss = "..smothloss..", Epoch = "..epoch)
  end
  if i % 200 == 0 then
    print("Saving network state..")
    torch.save('epoch'..epoch..'loss'..smothloss..'.net', rnn)
    print('done')
  end
  p = p+rho
end

-- save the model


-- function sample(seed, N)
--    local samples = torch.zeros(N, inputDimension)
--    samples[1] = rnn:forward(seed)
--    for i=2,N do
--       samples[i] = rnn:forward(samples[i-1])
--    end
--    return samples
-- end


-- --local seed = torch.rand(inputDimension) -- X[1]
-- local seed = X[p]
-- local samples = sample(seed, 400)
-- -- print(samples)
-- CSV.write(samples,'samples.csv')

