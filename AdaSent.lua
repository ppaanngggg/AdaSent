require 'nn'
require 'rnn'
require 'pnn'
-- require 'cutorch'
-- require 'cunn'


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

new_dataset = {}
for i = 1,200 do
    new_dataset[i] = dataset[i]
end
dataset = new_dataset
print("dataset finish")

use_batch = true

if use_batch == true then
    print('batch')
    batch_dataset = nn.pnn.datasetBatch(dataset, 25)
    -- batch_dataset = pnn.recursiveCuda(batch_dataset)
    function batch_dataset:size() return #batch_dataset end

    smodel = nn.BatchTable(model)
    -- smodel:cuda()
    -- ptable = nn.BatchTableCudaParallel(model, {1,2})

    cri = nn.BatchTableCriterion(nn.CrossEntropyCriterion())
    -- cri:cuda()
    --
    trainer = nn.StochasticGradient(smodel, cri)
    -- trainer = nn.StochasticGradient(ptable, cri)
    trainer.learningRate = 0.01
    trainer.maxIteration = 1
    begin_time = torch.tic()
    trainer:train(batch_dataset)
    print(torch.tic() - begin_time)

    for i = 1,5 do
        trainer = nn.StochasticGradient(smodel, cri)
        -- trainer = nn.StochasticGradient(ptable, cri)
        trainer.learningRate = 0.01
        trainer.maxIteration = 1
        begin_time = torch.tic()
        trainer:train(batch_dataset)
        print(torch.tic() - begin_time)
    end
else

    print('no batch')

    -- dataset = pnn.recursiveCuda(dataset)
    function dataset:size() return #dataset end

    -- model:cuda()
    criterion = nn.CrossEntropyCriterion()
    -- criterion:cuda()
    --
    trainer = nn.StochasticGradient(model, criterion)
    trainer.learningRate = 0.01
    trainer.maxIteration = 1
    begin_time = torch.tic()
    trainer:train(dataset)
    print(torch.tic() - begin_time)

    for i = 1,5 do
        trainer = nn.StochasticGradient(model, criterion)
        trainer.learningRate = 0.01
        trainer.maxIteration = 1
        begin_time = torch.tic()
        trainer:train(dataset)
        print(torch.tic() - begin_time)
    end
end
