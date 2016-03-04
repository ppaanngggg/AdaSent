--[[
	This file implements table parallelism for Torch modules.

	The same model is replicated on multiple GPUs. The input is split, typeically
	into smaller mini-batch.
]]--

local TableParallelTable, parent = torch.class('nn.TableParallelTable', 'nn.Container')

local threads = require 'threads'

function TableParallelTable:__init(module, gpuTable)
	parent.__init(self)

	self.module = module
	self.modules[1] = self.module

	self.gpuTable = gpuTable
	self.pool = threads.Threads(
		#gpuTable,
		function(id)
			require 'nn'
			require 'rnn'
			require 'cutorch'
			require 'cunn'
			require 'Cycle'
			require 'Slice'
			require 'SliceTable'
			print(gpuTable[id])
			cutorch.setDevice(gpuTable[id])
		end,
		function(id)
			function toCuda(input)
				if torch.type(input) == 'table' then
					local ret = {}
					for k,v in ipairs(input) do
						ret[k] = toCuda(v)
					end
					return ret
				else
					return input:cuda()
				end
			end
			function toFloat(input)
				if torch.type(input) == 'table' then
					local ret = {}
					for k,v in ipairs(input) do
						ret[k] = toFloat(v)
					end
					return ret
				else
					return input:float()
				end
			end
			tModule = module:clone():cuda()
		end
	)
	self.pool:specific(true)
end

function TableParallelTable:splitInput(input)
	local inputTable = {}
	local 
end

function TableParallelTable:updateOutput(input)
	self.output = {}
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function()
				print(cutorch.getDevice())
				local _input = toCuda(input[i])
				local _output = tModule:forward(_input)
				return toFloat(_output)
			end,
			function(tOutput)
				self.output[i] = tOutput
			end
		)
	end
	self.pool:synchronize()
	print(self.output)
	return self.output
end

function TableParallelTable:updateGradInput(input, gradOutput)

end
