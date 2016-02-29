require 'nn'
require 'rnn'

require 'SliceTable'
require 'Slice'
require 'Cycle'

-- params
params = {
    WORDVEC_DIM = 4,
    HIDDEN_DIM = 5,
    WEIGHT_NUM = 3,
    CLASSIFY_HIDDEN_DIM = 10,
    CLASSIFY_OUTPUT_DIM = 20,
    GATE_HIDDEN_DIM = 15,
    GATE_OUTPUT_DIM = 1
}

input = torch.rand(15,4)
-- print(input)

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

model = nn.Sequential()
    :add(nn.ParallelTable()
        :add(
            nn.Sequential()
                :add(up_proj_module)
                :add(
                    nn.ConcatTable()
                        :add(nn.Identity())
                        :add(nn.Identity())
                )
        )
        :add(nn.Identity())
    )
    :add(nn.FlattenTable())
    :add(
        nn.ConcatTable()
            :add(nn.SelectTable(1))
            :add(
                nn.Sequential()
                    :add(nn.NarrowTable(2,2))
                    :add(nn.Cycle(layer_up_module))
            )
    )
    :add(nn.FlattenTable())
    :add(
            nn.Sequencer(
                nn.Sequential()
                    :add(nn.Max(1))
                    :add(nn.Reshape(1, params.HIDDEN_DIM))
            )
        )
    :add(nn.JoinTable(1))
    :add(
        nn.ConcatTable()
            :add(
                nn.Sequential()
                    :add(nn.Linear(params.HIDDEN_DIM, params.GATE_HIDDEN_DIM))
                    :add(nn.Tanh())
                    :add(nn.Linear(params.GATE_HIDDEN_DIM, params.GATE_OUTPUT_DIM))
                    :add(nn.Tanh())
                    :add(nn.Transpose({1,2}))
                    :add(nn.SoftMax())
            )
            :add(
                nn.Sequential()
                    :add(nn.Linear(params.HIDDEN_DIM, params.CLASSIFY_HIDDEN_DIM))
                    :add(nn.Tanh())
                    :add(nn.Linear(params.CLASSIFY_HIDDEN_DIM, params.CLASSIFY_OUTPUT_DIM))
                    :add(nn.Tanh())
                    :add(nn.SoftMax())
            )
    )
    :add(nn.MM())

actualInput = {input, torch.LongTensor{input:size()[1] - 1}}
output = model:forward(actualInput)
print(output)

gradOutput = torch.zeros(1, params.CLASSIFY_OUTPUT_DIM)
gradOutput[{1,1}] = 1
print(gradOutput)
gradInput = model:backward(actualInput, gradOutput)
print(gradInput[1])

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
