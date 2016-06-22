--[[
-- stores a log of various aspects of training for the purpose of visualization
 ]]

local json = require 'cjson'
local paths = require 'paths'

local function write_json(file, t)
    local filename = file .. '.json'
    local f = io.open(filename, 'w')
    f:write(json.encode(t))
    f:close()
end

local function load_json(file)
    local filename = file .. '.json'
    if not paths.filep(filename) then
        return nil
    end
    local f = io.open(filename, 'r')
    local contents = f:read('*a')
    f:close()
    return json.decode(contents)
end

local Log = torch.class("Log")

function Log:__init(name, hyperparams, saveDir, xLabel, saveFrequency)
    self.name = name
    self.xLabel = xLabel or "Iterations" -- name of the x axis, usually related to time / number of batches / epochs
    self.hyperparams = hyperparams
    self.saveLoc = paths.concat(saveDir, name)
    self.saveFrequency = saveFrequency or 0
    self.data = {}
    self.updatesCounter = 0

    if not paths.filep(self.saveLoc .. '.json') then
        write_json(self.saveLoc, {})
    end
    -- update index file
    local indexLoc = paths.concat(saveDir, 'index')

    local models = {}
    for f in paths.files(saveDir, '.json') do
        if f ~= "index.json" and f:sub(1,1) ~= '.' then
            table.insert(models, f:sub(1, -6))
        end
    end
    write_json(indexLoc, models)
end



--[[
--  adds the data point (x, ys), where ys is a dictionary of different statistics to keep track of
 ]]
function Log:update(ys, x)
    local x = x or self.updatesCounter

    for name, y in pairs(ys) do
        local point = {x = x, y = y }
        -- if dataset doesn't exist, creat eit
        if not self.data[name] then
            self.data[name] = {}
        end
         -- add the point to it
        table.insert(self.data[name], point)
    end

    self.updatesCounter = self.updatesCounter + 1

    if self.saveFrequency > 0 and self.updatesCounter % self.saveFrequency == 0 then
        self:save()
    end
end

--[[
-- Saves all the data as saveDir/name.json, along with the given statistics
 ]]
function Log:save(stats)
    local stats = stats or {}
    write_json(self.saveLoc, {
        name = self.name,
        xLabel = self.xLabel,
        hyperparams = self.hyperparams,
        data = self.data,
        stats = stats
    });
end