require 'nn'
require 'dpnn'
require 'FixedLookupTable'

local HypernymScore, parent = torch.class('nn.HypernymScore', 'nn.Sequential')

function HypernymScore:__init(params, word_embeddings)
    parent.__init(self)
    local we = word_embeddings
    local lookup = nn.LookupTable(we:size(1), params.D_embedding)
    --lookup.weight:resizeAs(we):copy(we)

    local embedding = nn.Sequential():add(lookup)
    local embedding2 = embedding:sharedClone()

    -- self: takes two input words, outputs a probability that the first is a hypernym of the second
    self:add(nn.ParallelTable():add(embedding):add(embedding2))
    if params.symmetric then
        self:add(nn.CosineDistance())
        self:add(nn.AddConstant(1))
        self:add(nn.MulConstant(0.5))
    else
        self:add(nn.CSubTable())
        self:add(nn.ReLU()) -- i.e. max(0, x)
        self:add(nn.Mean(2))
    end

    if USE_CUDA then
        self:cuda()
        --reshare parameters
        embedding:share(embedding2, 'weight', 'bias', 'gradWeight', 'gradBias')
    end

    self.lookupModule = lookup
end

