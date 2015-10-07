local dataset = require 'Dataset'

local argparse = require 'argparse'

parser = argparse('Train a WordNet completion model')
parser:option '-d' :description 'dimensionality of embedding space' :convert(tonumber)
parser:option '--epochs' :description 'number of epochs to train for ' :default "1" :convert(tonumber)
parser:option '--lr' :description 'learning rate' :default "0.01" :convert(tonumber)

local args = parser:parse()

local hyperparams = {
    D_embedding = args.d,
    D_word = dataset.D_word
}


require 'optim'
require 'HypernymScore'
local config = { learningRate = args.lr }

local hypernymNet = nn.HypernymScore(hyperparams)
local criterion = nn.BCECriterion()

local parameters, gradients = hypernymNet:getParameters()

local count = 1
while dataset.epoch <= args.epochs do
    print("Epoch " .. dataset.epoch .. " Batch " .. count)
    count = count + 1
    local function eval(x)
        hypernymNet:zeroGradParameters()
        local input, target = dataset.nextBatch()
        local probs = hypernymNet:forward(input)
        local err = criterion:forward(probs, target)
        print("Error: " .. err)
        --print("Norm of parameters: " .. torch.norm(parameters))
        local gProbs = criterion:backward(probs, target)
        local _ = hypernymNet:backward(input, gProbs)
        return err, gradients
    end

    optim.adam(eval, parameters, config)
end





