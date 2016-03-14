dofile('AdaSent.lua')

cutorch.setDevice(5)

dataset = torch.load('./test_dataset')
dataset = pnn.recursiveCuda(dataset)


model = torch.loadobj('model')
model:evaluate()
model:cuda()

correctNum = 0

for i = 1,#dataset do
    output = model:forward(dataset[i][1])
    maxIndex = 0
    maxValue = -1
    for j = 1,output:size()[2] do
        if output[1][j] > maxValue then
            maxValue = output[1][j]
            maxIndex = j
        end
    end
    print(maxIndex, dataset[i][2], output[1][dataset[i][2]])
    if maxIndex == dataset[i][2] then
        correctNum = correctNum + 1
    end
end

print(correctNum / #dataset * 100)
