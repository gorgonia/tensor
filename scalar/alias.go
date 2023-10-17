package scalar

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
)

type AP = tensor.AP
type Engine = internal.Engine

type DescWithStorage = tensor.DescWithStorage
type ConsOpt = tensor.ConsOpt
type DataOrder = internal.DataOrder

type MemoryFlag = internal.MemoryFlag

const (
	ColMajor = internal.ColMajor
)
