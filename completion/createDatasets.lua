require 'Dataset'

torch.manualSeed(1234)
local method = 'random'

local hdf5 = require 'hdf5'

local f = hdf5.open('dataset/wordnet.h5', 'r')
local word_embeddings = f:read('embeddings'):all()
local D_word = word_embeddings:size(2)
local in_w2v = f:read('in_w2v'):all():byte()
local hypernyms = f:read('hypernyms'):all():add(1) -- convert to 1-based indexing
local entity2word = f:read('entity2word'):all():add(1) -- convert to 1-based indexing
local slices = f:read('slices'):all():add(1)
slices[{{}, 2}]:add(-1)
local N_hypernyms = hypernyms:size(1)
f:close()
print("Loaded data")

-- initialize word embeddings that we don't have, randomly
local nonzero = in_w2v:eq(0):nonzero():view(-1)
print("Number of nonzero word embeddings: " .. nonzero:size(1))
local random = torch.randn(nonzero:size(1), D_word)
word_embeddings:indexCopy(1, nonzero, random)
-- normalize them to have unit norm
local norms = torch.norm(word_embeddings, 2, 2)
word_embeddings:cdiv(norms:view(-1, 1):expandAs(word_embeddings))

-----
-- split hypernyms into train, dev, test
-----

local splitSize = 4000

-- shuffle randomly
local order = torch.randperm(N_hypernyms):long()
local hypernyms = hypernyms:index(1, order)
print("Building sets ...")
local sets = {
    test = hypernyms:narrow(1, 1, splitSize),
    val1 = hypernyms:narrow(1, splitSize + 1, splitSize),
    val2 = hypernyms:narrow(1, splitSize*2 + 1, splitSize),
    train = hypernyms:narrow(1, splitSize*3+ 1, N_hypernyms - 3*splitSize)
}
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

torch.save('dataset/' .. method .. '.t7', datasets)
write_json('vis/static/' .. method .. '/hypernyms', datasets.train.hypernyms:totable())



