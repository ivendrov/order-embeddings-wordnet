local argparse = require 'argparse'

parser = argparse('Train a WordNet completion model')
parser:option '-d' :description 'dimensionality of embedding space' :default "10" :convert(tonumber)
parser:option '--epochs' :description 'number of epochs to train for ' :default "1" :convert(tonumber)
parser:option '--batchsize' :description 'size of minibatch to use' :default "1000" :convert(tonumber)
parser:option '--eval_freq' :description 'evaluation frequency' :default "100" :convert(tonumber)
parser:option '--lr' :description 'learning rate' :default "0.1" :convert(tonumber)
parser:option '--dataset' :description 'dataset to use' :default 'random'

local args = parser:parse()

require 'Dataset'
local datasets = torch.load('dataset/' .. args.dataset .. '.t7')
local train = datasets.train


local hyperparams = {
    D_embedding = args.d,
    D_word = train.D_word
}


require 'optim'
require 'HypernymScore'
local config = { learningRate = args.lr }

local hypernymNet = nn.HypernymScore(hyperparams)
local criterion = nn.BCECriterion()

----------------
-- EVALUATION --
----------------

-- returns optimal threshold, and classification at that threshold, for the given dataset
local function findOptimalThreshold(dataset, model)
    local input, target = dataset:all()
    local probs = model:forward(input)
    local sortedProbs, indices = torch.sort(probs, 1, true) -- sort in descending order
    local sortedTarget = target:index(1, indices)
    local Nneg = sortedTarget:eq(0):sum() -- number of negatives
    local tp = torch.cumsum(sortedTarget)
    local fp = torch.cumsum(sortedTarget:eq(0):double())
    local tn = fp:mul(-1):add(Nneg)
    local accuracies = tp:add(tn):div(sortedTarget:size(1))
    local bestAccuracy, i = torch.max(accuracies, 1)
    return sortedProbs[i[1]], bestAccuracy[1]
end

-- evaluate model at given threshold
local function evalClassification(dataset, model, threshold)
    local input, target = dataset:all()
    local probs = model:forward(input)

    local inferred = probs:ge(threshold)
    local accuracy = inferred:eq(target:byte()):double():mean()
    return accuracy
end



--------------
-- TRAINING --
--------------
local parameters, gradients = hypernymNet:getParameters()

local count = 1
while train.epoch <= args.epochs do
    count = count + 1
    local function eval(x)
        hypernymNet:zeroGradParameters()
        local input, target = train:minibatch(args.batchsize)
        local probs = hypernymNet:forward(input)
        local err = criterion:forward(probs, target)
        if count % 10 == 0 then
            print("Epoch " .. train.epoch .. " Batch " .. count .. " Error " .. err)
        end
        local gProbs = criterion:backward(probs, target)
        local _ = hypernymNet:backward(input, gProbs)
        return err, gradients
    end

    optim.adam(eval, parameters, config)

    if count % args.eval_freq == 0 then
        --print("Evaluating:")
        local threshold, accuracy = findOptimalThreshold(datasets.val1, hypernymNet)
        --print("Best accuracy " .. accuracy .. " at threshold " .. threshold)
        print("Accuracy " .. evalClassification(datasets.val2, hypernymNet, threshold))
    end
end





