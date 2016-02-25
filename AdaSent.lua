require 'nn'
require 'rnn'

require 'SliceTable'
require 'Slice'

-- params
params = {
WORDVEC_DIM = 4,
HIDDEN_DIM = 5,
WEIGHT_NUM = 3,
}

input = torch.rand(10,4)
-- print(input)

model = nn.Sequential()

-- project from word vector space into higher sentence space
up_proj_module = nn.Linear(params.WORDVEC_DIM, params.HIDDEN_DIM, false)


softmax_module = nn.Sequential()
    :add(
        nn.ConcatTable()
            :add(
                nn.Sequential()
                    :add(nn.Slice(1, -2))
                    :add(nn.Linear(params.HIDDEN_DIM, params.WEIGHT_NUM, false))
            )
            :add(
                nn.Sequential()
                    :add(nn.Slice(2, -1))
                    :add(nn.Linear(params.HIDDEN_DIM, params.WEIGHT_NUM, false))
            )
    )
    :add(nn.CAddTable())
    :add(nn.Add(params.WEIGHT_NUM))
    :add(nn.SoftMax())

h_left_module = nn.Slice(1, -2)

h_right_module = nn.Slice(2, -1)

h_tilde_module = nn.Sequential()
    :add(
        nn.ConcatTable()
            :add(
                nn.Sequential()
                    :add(nn.Slice(1, -2))
                    :add(nn.Linear(params.HIDDEN_DIM, params.HIDDEN_DIM, false))
            )
            :add(
                nn.Sequential()
                    :add(nn.Slice(2, -1))
                    :add(nn.Linear(params.HIDDEN_DIM, params.HIDDEN_DIM, false))
            )
    )
    :add(nn.CAddTable())
    :add(nn.Add(params.HIDDEN_DIM))
    :add(nn.Tanh())

layer_up_module = nn.Sequential()
    :add(
        nn.ConcatTable()
            :add(softmax_module)
            :add(nn.ConcatTable()
                :add(h_left_module)
                :add(h_right_module)
                :add(h_tilde_module)
            )
    )
    :add(nn.MixtureTable())

model:add(up_proj_module)
model:add(
    nn.Repeater(layer_up_module, 2)
)


output = model:forward(input)
print(output)

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
