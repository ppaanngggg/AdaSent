require 'nn'
require 'rnn'
require 'Cycle'
--require 'cutorch'
require 'TableParallelTable.lua'

add = nn.Add(5)

model = nn.TableParallelTable(add, {10,11})

input = {}
for i = 1,20 do
	input[i] = torch.rand(5)
end

model:forward(input)

--add = nn.Add(5)
--add.bias = torch.ones(5)

--input = torch.zeros(5)

--cycle = nn.Cycle(add)
--actualInput = {input, torch.LongTensor{2}}
--output = cycle:forward(actualInput)
--gradOutput = {torch.ones(5) * 1.2, torch.ones(5) * 2.5}
--gradInput = cycle:backward({input,2}, gradOutput)
--print(cycle.gradInput)
--print(add.gradBias)
