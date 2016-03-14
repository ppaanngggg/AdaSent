dofile('./AdaSent.lua')


dataset = torch.load('./train_dataset')

dataList = {}
for i = 1,#dataset do
    if dataList[dataset[i][2]] then
        dataList[dataset[i][2]] = dataList[dataset[i][2]] + 1
    else
        dataList[dataset[i][2]] = 1
    end
end

for i = 1,#dataList do
    dataList[i] = 50 / dataList[i]
end

model:training()
criterion = nn.CrossEntropyCriterion(torch.Tensor(dataList))
criterion.sizeAverage = false

local state = {
    learningRate = 0.01,
    learningRateDecay = 0.0002,
    momentum = 0.9,
    wd = 16
}
local trainer = nn.MultiGPUTrainer(model, criterion, optim.sgd, state, {1,2,3,4})
trainer:train(dataset, 80, 200)
