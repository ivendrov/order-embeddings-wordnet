local argparse = require 'argparse'

parser = argparse('Train a WordNet completion model')
parser:option '-d' :description 'dimensionality of embedding space' :default "100" :convert(tonumber)
parser:option '--epochs' :description 'number of epochs to train for ' :default "1" :convert(tonumber)
parser:option '--batchsize' :description 'size of minibatch to use' :default "1000" :convert(tonumber)
parser:option '--eval_freq' :description 'evaluation frequency' :default "100" :convert(tonumber)
parser:option '--lr' :description 'learning rate' :default "0.1" :convert(tonumber)
parser:option '--dataset' :description 'dataset to use' :default 'random'

USE_CUDA = true
if USE_CUDA then
    require 'cutorch'
    require 'cunn'
end

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

local function cudify(input, target)
    if USE_CUDA then
        return {input[1]:cuda(), input[2]:cuda()}, target
    else
        return input, target
    end
end

-- returns optimal threshold, and classification at that threshold, for the given dataset
local function findOptimalThreshold(dataset, model)
    local input, target = cudify(dataset:all())
    local probs = model:forward(input):double()
    local sortedProbs, indices = torch.sort(probs, 1, true) -- sort in descending order
    local sortedTarget = target:index(1, indices:long())
    local tp = torch.cumsum(sortedTarget)
    local invSortedTarget = torch.eq(sortedTarget, 0):double()
    local Nneg = invSortedTarget:sum() -- number of negatives
    local fp = torch.cumsum(invSortedTarget)
    local tn = fp:mul(-1):add(Nneg)
    local accuracies = tp:add(tn):div(sortedTarget:size(1))
    local bestAccuracy, i = torch.max(accuracies, 1)
    return sortedProbs[i[1]], bestAccuracy[1]
end

-- evaluate model at given threshold
local function evalClassification(dataset, model, threshold)
    local input, target = cudify(dataset:all())
    local probs = model:forward(input):double()

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
        local input, target = cudify(train:minibatch(args.batchsize))
        local probs = hypernymNet:forward(input):double()
        local err = criterion:forward(probs, target)
        if count % 10 == 0 then
            print("Epoch " .. train.epoch .. " Batch " .. count .. " Error " .. err)
        end
        local gProbs = criterion:backward(probs, target):cuda()
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





