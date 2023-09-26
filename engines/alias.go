package stdeng

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
)

type SliceRange = internal.SliceRange

type Memory = tensor.Memory
type MemoryFlag = internal.MemoryFlag
type FuncOpt = tensor.FuncOpt
type Engine = tensor.Engine
type DataOrder = internal.DataOrder

type Desc = tensor.Desc
type DescWithStorage = tensor.DescWithStorage

type Iterator = tensor.Iterator

type Memsetter = tensor.Memsetter

/* Constraints */
type OrderedNum = internal.OrderedNum
type Num = internal.Num
type Addable = internal.Addable

type Option = tensor.Option
type ConsOpt = tensor.ConsOpt

const AllAxes = internal.AllAxes

func WithBacking(backing any) ConsOpt             { return tensor.WithBacking(backing) }
func WithShape(shp ...int) ConsOpt                { return tensor.WithShape(shp...) }
func ParseFuncOpts(opts ...FuncOpt) tensor.Option { return tensor.ParseFuncOpts(opts...) }
