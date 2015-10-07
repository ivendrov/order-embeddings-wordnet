require 'nn'
require 'dpnn'

local HypernymScore, parent = torch.class('nn.HypernymScore', 'nn.Sequential')

function HypernymScore:__init(params)
    parent.__init(self)
    local embedding = nn.Sequential():add(nn.Linear(params.D_word, params.D_embedding)):add(nn.Sigmoid())

    -- self: takes two input words, outputs a probability that the first is a hypernym of the second
    self:add(nn.ParallelTable():add(embedding):add(embedding:sharedClone()))
    self:add(nn.CSubTable())
    self:add(nn.ReLU()) -- i.e. max(0, x)
    -- take L2 norm of difference
    self:add(nn.Power(2))
    self:add(nn.Sum(2))
    self:add(nn.Sqrt())
    -- turn into probability
    --self:add(nn.Mul())
    self:add(nn.MulConstant(-1))
    self:add(nn.Exp())
end

