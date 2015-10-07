local Dataset = torch.class('Dataset')


local function genNegatives(N, N_words, method)
    if method == 'random' then
        return torch.rand(N, 2):mul(N_words):ceil():cmax(1):long()
    end
end

-- dataset creation
function Dataset:__init(embeddings, hypernyms, method)
    self.method = method
    self.hypernyms = hypernyms
    local N_hypernyms = hypernyms:size(1)
    local N_words = embeddings:size(1)
    self.genNegatives = function() return genNegatives(N_hypernyms, N_words, method) end

    self:regenNegatives()
    self.epoch = 0
end

function Dataset:regenNegatives()
    local negatives = self.genNegatives()
    local all_hypernyms = torch.cat(self.hypernyms, negatives, 1)
    self.hyper = all_hypernyms[{{}, 2}]
    self.hypo = all_hypernyms[{{}, 1}]
    self.target = torch.cat(torch.ones(self.hypernyms:size(1)), torch.zeros(negatives:size(1)))
end



function Dataset:size()
    return self.target:size(1)
end

local function createInput(embeddings, indices)
    local hyper = indices[1]
    local hypo = indices[2]
    return {embeddings:index(1, hyper), embeddings:index(1, hypo)}
end

function Dataset:minibatch(size)
    if not self.s or self.s + size - 1 > self:size() then
        -- new epoch; randomly shuffle dataset
        self.order = torch.randperm(self:size()):long()
        self.s = 1
        self.epoch = self.epoch + 1
        -- regenerate negatives
        self:regenNegatives()
    end

    local s = self.s
    local e = s + size - 1

    local indices = self.order[{{s,e}}]
    local hyper = self.hyper:index(1, indices)
    local hypo = self.hypo:index(1, indices)
    local target = self.target:index(1, indices)

    self.s = e + 1

    return {hyper, hypo}, target
end

function Dataset:all()
    return {self.hyper, self.hypo}, self.target
end