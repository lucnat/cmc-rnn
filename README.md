## Classical Music Composition with Recurrent Neural Networks.


Using the code in `char-rnn` we trained the network with the Koran and with "The Origin Of Species". `/char-rnn-results` contains sample text of 100'000 characters generated from those networks as well as the corresponding checkpoints that allows you to generate more text. 

## Run the Code yourself

Navigate to `csv-rnn` and you will find there among others the files `train.lua` and `sample.lua`. You can now train the 
net by running

`th train.lua -csv_file bounce.csv`

The parameters (network size, amount of layers etc.) can be set within the code. The code will write
state files of the network's state every 200 iterations. In order to sample from the model, simply do

`th sample.lua -state_file epoch24loss0.01362422014985.net -input_dimension 8 -length 1000 > out.csv`

This is going to write a csv file of 1000 generated vectors of size 8 (this has to be the size the 
network has actually been trained with). 
Enjoy the ride!

## Links

Paper (Work in Progress): https://www.overleaf.com/read/cnkdtvbtfsyd

Research Plan: https://www.overleaf.com/3808219jvshrb

Character language models: https://github.com/karpathy/char-rnn
