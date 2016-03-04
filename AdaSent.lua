require 'nn'
require 'rnn'
--require 'cutorch'
--require 'cunn'

require 'SliceTable'
require 'Slice'
require 'Cycle'
require 'TableParallelTable'

-- params
params = {
    WORDVEC_DIM = 300,
    HIDDEN_DIM = 500,
    WEIGHT_NUM = 3,
    CLASSIFY_HIDDEN_DIM = 200,
    CLASSIFY_OUTPUT_DIM = 50,
    GATE_HIDDEN_DIM = 200,
    GATE_OUTPUT_DIM = 1
}

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
            )
    )
    :add(nn.MM())

dataset = torch.load('dataset')
function dataset:size() return #dataset end
print('finish load dataset')

input = {}
gradOutput = {}
for i = 1, 10 do
	input[i] = dataset[i][1]
    gradOutput[i] = torch.rand(1, 50)
end

model:forward(input[1])
model:backward(input[1], gradOutput[1])

smodel = nn.Sequencer(model)
smodel:training()
soutput = smodel:forward(input)
sgradinput = smodel:backward(input, gradOutput)
print(sgradinput[1])

ptable = nn.TableParallelTable(model, {1,2,3})
poutput = ptable:forward(input)
pgradinput = ptable:backward(input, gradOutput)
print(pgradinput[1])

--dataset = torch.load('dataset')
--for i = 1,#dataset do
    --dataset[i][1][1] = dataset[i][1][1]:cuda()
--end
--function dataset:size() return #dataset end
--print('finish load dataset')

--model:cuda()
--criterion = nn.CrossEntropyCriterion():cuda()

--trainer = nn.StochasticGradient(model, criterion)
--trainer.learningRate = 0.01
--trainer.maxIteration = 1000
--trainer:train(dataset)

--output = model:forward(dataset[1][1])
--print(output)
--print(dataset[1][2])


-- actualInput = {torch.rand(10,300), torch.LongTensor{9}}
-- output = model:forward(actualInput)
-- print(output)

-- criterion = nn.CrossEntropyCriterion()

-- local err = criterion:forward(output, 20)
-- print(err)
-- model:zeroGradParameters()
-- local t = criterion:backward(output, 20)
-- print(t)
-- model:backward(actualInput, t)
-- model:updateParameters(1)
