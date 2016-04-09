# Instructions for Mac OSX
### 1.Install Lua
The easiest way to to get Lua is through `brew`. If you don't have brew installed, get it by pasting the following line into the terminal:

`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

Then, install Lua by typing `brew install lua` and `brew update`. Type `lua` to start the Lua console. 

### 2. Install Torch
In terminal, run the commands
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```
This installs torch into your home folder. You now might want to add the torch bin folder to your `$PATH`. Do this by adding the path `/Users/zeus/torch/install/bin`to your `.bash_profile` file. You can find the `.bash_profile`in your homefolder. If it doesn't exist yet, create it (replace user zeus by your actual user from now on):

```
nano /Users/zeus/.bash_profile
```
Now add the following line to this file to concat the new path:
```
export PATH="/Users/zeus/torch/install/bin:$PATH"
```
Save the file with `ctrl+O`, press enter to save, and exit nano with `ctrl+X`. Restart your terminal. Check if the terminal can find the binaries by typing `th` in your console. If the torch console starts, you are golden. 

## 3. Train and sample

Put your input file named `input.txt` into the `data`folder, then run

`th train.lua -data_dir data/ -rnn_size 512 -num_layers 2 -dropout 0.5 -gpuid -1`

This trains an RNN with 2 hidden layers and 512 units each layer on the `input.txt`file using stochastic batch gradient with cpu. All 1000 iterations, checkpoints are being written to the `cv`folder. The checkpoints can then be used to generatate a text of 2000 characters: 

`th sample.lua cv/some_checkpoint.t7 -length 2000 -gpuid -1`

For detailed instructions, see https://github.com/karpathy/char-rnn
