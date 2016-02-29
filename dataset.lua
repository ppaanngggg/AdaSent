mongorover = require("mongorover")
require 'torch'

local client = mongorover.MongoClient.new("mongodb://127.0.0.1:27017/")
local db = client:getDatabase("TREC")
local coll = db:getCollection("train")
dataset = {}
index = 1
for d in coll:find({}) do
    input = torch.Tensor(d['vecs'])
    label = d['label']
    actualInput = {input, torch.LongTensor{input:size()[1] - 1}}
    dataset[index] = {actualInput, label}
    index = index + 1
end
function dataset:size() return #dataset end
