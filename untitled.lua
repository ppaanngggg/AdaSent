require 'nn'
require 'rnn'

require 'SliceTable'
require 'Slice'

-- params
params = {
WORDVEC_DIM = 3,
HIDDEN_DIM = 5
}

input = torch.rand(10,3)
print(input)
-- up_proj_module = nn.Linear(params.WORDVEC_DIM, params.HIDDEN_DIM, false)
-- print(up_proj_module)
-- output = up_proj_module:forward(input)
--
-- layer = nn.ConcatTable()
--
softmax = nn.ConcatTable()
softmax:add(
    nn.Sequential()
        :add(nn.Slice(1, -2))
        :add(nn.Linear(params.HIDDEN_DIM, params.HIDDEN_DIM, false))
)
softmax:add(
    nn.Sequential()
        :add(nn.Slice(2, -1))
        :add(nn.Linear(params.HIDDEN_DIM, params.HIDDEN_DIM, false))
)
print(softmax)

-- local mongorover = require("mongorover")
-- local client = mongorover.MongoClient.new("mongodb://127.0.0.1:27017/")
-- local database_names = client:getDatabaseNames()
-- for k,v in ipairs(database_names) do
--     print(k,v)
-- end
-- local db = client:getDatabase("SeekingAlpha")
-- local collection_names = db:getCollectionNames()
-- for k,v in ipairs(collection_names) do
--     print(k,v)
-- end
-- local coll = db:getCollection("AAPL_vec")
-- for d in coll:find({}) do
--     print(d.date)
-- end
