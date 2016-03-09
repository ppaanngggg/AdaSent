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
    dataList[i] = #dataset / dataList[i]
end

batch_dataset = nn.pnn.datasetBatch(dataset, 8)
function batch_dataset:size() return #batch_dataset end

gpuTable = {1,2,3,4}
smodel = nn.BatchTable(model)
criterion = nn.BatchTableCriterion(nn.CrossEntropyCriterion(torch.Tensor(dataList)))

adam_state = {}

trainer = nn.MultiGPUTrainer(smodel, criterion, optim.adam, adam_state, gpuTable)
trainer:train(batch_dataset, 100)

torch.saveobj('smodel', smodel)
torch.saveobj('adam_state', adam_state)
