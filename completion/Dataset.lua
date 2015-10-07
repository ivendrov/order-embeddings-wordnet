local dataset = {}

local hdf5 = require 'hdf5'

local f = hdf5.open('wordnet.h5', 'r')
local word_embeddings = f:read('embeddings'):all()
dataset.D_word = word_embeddings:size(2)
local in_w2v = f:read('in_w2v'):all():byte()
local hypernyms = f:read('hypernyms'):all():add(1) -- convert to 1-based indexing
local N_hypernyms = hypernyms:size(1)
f:close()
print("Loaded data")

-- initialize word embeddings that we don't have, randomly
local nonzero = in_w2v:eq(0):nonzero():view(-1)
print("Number of nonzero word embeddings: " .. nonzero:size(1))
local random = torch.randn(nonzero:size(1), dataset.D_word)
word_embeddings:indexCopy(1, nonzero, random)
-- normalize them to have unit norm
local norms = torch.norm(word_embeddings, 2, 2)
word_embeddings:cdiv(norms:view(-1, 1):expandAs(word_embeddings))


-- returns the input and target
dataset.epoch = 1
local s = 1
local batchsize = 1000
local order = torch.randperm(N_hypernyms):long()
function dataset.nextBatch()
    local e = math.min(s + batchsize - 1, N_hypernyms)
    local hypernyms = hypernyms:index(1, order[{{s,e}}])
    local hypo = word_embeddings:index(1, hypernyms[{{}, 1}])
    local hyper = word_embeddings:index(1, hypernyms[{{}, 2}])
    local target = torch.ones(hypo:size(1))
    -- TODO add negative examples
    s = e + 1
    if s > N_hypernyms then
        order = torch.randperm(N_hypernyms):long()
        s = 1
        dataset.epoch = dataset.epoch + 1
    end
    return {hyper, hypo}, target
end





return dataset