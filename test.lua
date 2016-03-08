require 'nn'
require 'pnn'

-- linear = nn.Linear(5,3)
--
-- model = nn.TableParallelTable(linear, {1,2,3})
-- params, gradParams = model:parameters()
--
-- input = {}
-- for i = 1,20 do
-- 	input[i] = torch.rand(5)
-- end
--
-- gradOutput = {}
-- for i = 1,20 do
-- 	gradOutput[i] = torch.rand(3)
-- end
--
-- model:forward(input)
-- model:backward(input, gradOutput)
-- model:updateParameters(1)
-- model:zeroGradParameters()

add = nn.Add(5)
add.bias = torch.ones(5)

input = torch.zeros(5)

cycle = nn.Cycle(add)
actualInput = {input, torch.LongTensor{2}}
output = cycle:forward(actualInput)

for k,v in ipairs(output) do
	print(v)
end

gradOutput = {torch.ones(5) * 1.2, torch.ones(5) * 2.5}
gradInput = cycle:backward({input,2}, gradOutput)
--print(cycle.gradInput)
--print(add.gradBias)
