require 'Dataset'

torch.manualSeed(1234)
local method = 'random'

local hdf5 = require 'hdf5'

local f = hdf5.open('dataset/wordnet.h5', 'r')
local word_embeddings = f:read('embeddings'):all()
local D_word = word_embeddings:size(2)
local in_w2v = f:read('in_w2v'):all():byte()
local originalHypernyms = f:read('hypernyms'):all():add(1) -- convert to 1-based indexing
local entity2word = f:read('entity2word'):all():add(1) -- convert to 1-based indexing
local slices = f:read('slices'):all():add(1)
slices[{{}, 2}]:add(-1)
f:close()
print("Loaded data")




local graph = require 'Graph'

-----
-- split hypernyms into train, dev, test
-----
for _, overfit in ipairs{true, false} do
    for _, hypernymType in ipairs{'trans', 'notrans'} do
        local methodName = method
        local hypernyms = originalHypernyms
        if hypernymType == 'trans' then
            hypernyms = graph.transitiveClosure(hypernyms)
            methodName = methodName .. '_trans'
        end

        local N_hypernyms = hypernyms:size(1)
        local splitSize = 4000

        -- shuffle randomly
        torch.manualSeed(1)
        local order = torch.randperm(N_hypernyms):long()
        local hypernyms = hypernyms:index(1, order)
        print("Building sets ...")

        local sets
        if overfit then
            methodName = methodName .. '_overfit'
            sets = {
                train = hypernyms,
                val1 = hypernyms,
                val2 = hypernyms
            }
        else
            sets = {
                test = hypernyms:narrow(1, 1, splitSize),
                val1 = hypernyms:narrow(1, splitSize + 1, splitSize),
                val2 = hypernyms:narrow(1, splitSize*2 + 1, splitSize),
                train = hypernyms:narrow(1, splitSize*3+ 1, N_hypernyms - 3*splitSize)
            }
        end
        print("Done. Building Datasets ...")
        local datasets = {}
        for name, hnyms in pairs(sets) do
            datasets[name] = Dataset(slices:size(1), hnyms, method)
        end

        datasets.word_embeddings = word_embeddings
        datasets.entity2word = entity2word
        datasets.slices = slices


        -- save visualization info
        local paths = require 'paths'
        local json = require 'cjson'
        local function write_json(file, t)
            local filename = file .. '.json'
            paths.mkdir(paths.dirname(filename))
            local f = io.open(filename, 'w')
            f:write(json.encode(t))
            f:close()
        end

        torch.save('dataset/' .. methodName .. '.t7', datasets)
        write_json('vis/static/' .. methodName .. '/hypernyms', datasets.train.hypernyms:totable())
    end
end



