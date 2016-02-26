require 'torch'
require 'nn'
require 'rnn'
require 'Cycle'

input = torch.Tensor(5)
print(input)

add = nn.Add(5)
add.bias = torch.ones(5)
print(add.bias)

cycle = nn.Cycle(add)
print(cycle)
output = cycle:forward({input, 2})
for k,v in ipairs(output) do
    print(v)
end

gradOutput = {torch.ones(5) * 1.2, torch.ones(5) * 2.5}
for k,v in ipairs(gradOutput) do
    print(v)
end
gradInput = cycle:backward({input, 2}, gradOutput)
print(add.gradBias)
