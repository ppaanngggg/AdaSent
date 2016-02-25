------------------------------------------------------------------------
--[[ VariRepeater ]]--
-- Encapsulates an AbstractRecurrent instance (rnn) which is repeatedly
-- presented with the same input for rho(variable) time steps.
-- The output is a table of rho outputs of the rnn.
------------------------------------------------------------------------
assert(not nn.VariRepeater, "update nnx package : luarocks install nnx")
local VariRepeater, parent = torch.class('nn.VariRepeater', 'nn.AbstractSequencer')

function VariRepeater:__init(module)
    parent.__init(self)

    self.module = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(module) or module

    self.modules[1] = self.module
    self.output = {}
end

function VariRepeater:updateOutput(input)
    assert(torch.type(input) == 'table', "expecting input table")
    local rho = input[2]
    assert(torch.type(rho) == 'number', "expecting number value for arg 2")
    self.rho = rho
    self.module = self.module or self.rnn -- backwards compatibility
    self.module:maxBPTTstep(rho) -- hijack rho (max number of time-steps for backprop)

    self.module:forget()

    -- TODO make copy outputs optional
    for step=1,self.rho do
        if step == 1 then
            self.output[step] = nn.rnn.recursiveCopy(
                self.output[step],
                self.module:updateOutput(input[1])
            )
        else
            self.output[step] = nn.rnn.recursiveCopy(
                self.output[step],
                self.module:updateOutput(self.output[step-1])
            )
        end
    end
    return self.output
end

function VariRepeater:updateGradInput(input, gradOutput)
    assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
    assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
    assert(#gradOutput == self.rho, "gradOutput should have rho elements")

    -- back-propagate through time (BPTT)
    for step=self.rho,1,-1 do
        local gradInput
        if step == 1 then
            gradInput = self.module:updateGradInput(input[1], gradOutput[step])
        else
            gradInput = self.module:updateGradInput(self.output[step - 1])
        end
        if step == self.rho then
            self.gradInput = nn.rnn.recursiveCopy(self.gradInput, gradInput)
        else
            nn.rnn.recursiveAdd(self.gradInput, gradInput)
        end
    end

    return self.gradInput
end

function VariRepeater:accGradParameters(input, gradOutput, scale)
    assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
    assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
    assert(#gradOutput == self.rho, "gradOutput should have rho elements")

    -- back-propagate through time (BPTT)
    for step=self.rho,1,-1 do
        if step == 1 then
            self.module:accGradParameters(self.output[step - 1], gradOutput[step], scale)
        else
            self.module:accGradParameters(input[1], gradOutput[step], scale)
        end
    end

end

function VariRepeater:maxBPTTstep(rho)
    self.rho = rho
    self.module:maxBPTTstep(rho)
end

function VariRepeater:accUpdateGradParameters(input, gradOutput, lr)
    assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
    assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
    assert(#gradOutput == self.rho, "gradOutput should have rho elements")

    -- back-propagate through time (BPTT)
    for step=self.rho,1,-1 do
        if step == 1 then
            self.module:accUpdateGradParameters(self.output[step - 1], gradOutput[step], lr)
        else
            self.module:accUpdateGradParameters(input[1], gradOutput[step], lr)
        end
    end
end

function VariRepeater:__tostring__()
    local tab = '  '
    local line = '\n'
    local str = torch.type(self) .. ' {' .. line
    str = str .. tab .. '[  input,  output(1),...,output(-1)  ]'.. line
    str = str .. tab .. '     V         V             V     '.. line
    str = str .. tab .. tostring(self.modules[1]):gsub(line, line .. tab) .. line
    str = str .. tab .. '     V         V             V     '.. line
    str = str .. tab .. '[output(1),output(2),...,output('..self.rho..')]' .. line
    str = str .. '}'
    return str
end
