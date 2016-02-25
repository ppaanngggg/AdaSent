require 'nn'
require 'rnn'
require 'VariRepeater'

input = torch.Tensor(5)
print(input)

add = nn.Add(5)
print(add.bias)

reap = nn.VariRepeater(add)
output = reap:forward({input, 2})
for k,v in ipairs(output) do
    print(v)
end
