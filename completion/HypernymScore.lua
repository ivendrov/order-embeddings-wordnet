require 'nn'
require 'dpnn'
require 'FixedLookupTable'

local HypernymScore, parent = torch.class('nn.HypernymScore', 'nn.Sequential')

function HypernymScore:__init(params)
    parent.__init(self)
    local we = params.word_embeddings
    local lookup = nn.LookupTable(we:size(1),10)
    --lookup.weight:resizeAs(we):copy(we)

    local embedding = nn.Sequential():add(lookup):add(nn.Linear(10, params.D_embedding)):add(nn.Abs())
    local embedding2 = embedding:sharedClone()

    -- self: takes two input words, outputs a probability that the first is a hypernym of the second
    self:add(nn.ParallelTable():add(embedding):add(embedding2))
    self:add(nn.CSubTable())
    self:add(nn.ReLU()) -- i.e. max(0, x)
    -- take L2 norm of difference
    --self:add(nn.Power(2))
    self:add(nn.Mean(2))
    -- turn into probability
    --self:add(nn.Mul())
    self:add(nn.MulConstant(-1))
    self:add(nn.Exp())

    if USE_CUDA then
        self:cuda()
        --reshare parameters
        embedding:share(embedding2, 'weight', 'bias', 'gradWeight', 'gradBias')
    end
end

