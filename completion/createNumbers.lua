require 'Dataset'

torch.manualSeed(1234)
local method = 'numbers'



local nEntities = 1000

local hypernyms = {}
local negatives = {}
for i = 1, nEntities do
    for j = i+1, nEntities do
        if j % i == 0 then
            table.insert(hypernyms, {j, i})
        else
            table.insert(negatives, {j, i})
        end
    end
end

hypernyms = torch.LongTensor(hypernyms)
negatives = torch.LongTensor(negatives)

local N_hypernyms = hypernyms:size(1)
print("Number of hypernyms " .. N_hypernyms)

-----
-- split hypernyms into train, dev, test
-----

local splitSize = 50

-- shuffle randomly
local order = torch.randperm(N_hypernyms):long()
local hypernyms = hypernyms:index(1, order)
print("Building sets ...")
local sets = {
    val1 = hypernyms,
    val2 = hypernyms,
    train = hypernyms
}
print("Done. Building Datasets ...")
local datasets = {}
for name, hnyms in pairs(sets) do
    datasets[name] = Dataset(nEntities, hnyms, method, negatives)
end

datasets.slices = torch.ones(nEntities)


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

local names = {}
for i = 1, nEntities do
    table.insert(names, tostring(i))
end

torch.save('dataset/' .. method .. '.t7', datasets)
write_json('vis/static/' .. method .. '/hypernyms', datasets.train.hypernyms:totable())
write_json('vis/static/' .. method .. '/synset_names', names)

