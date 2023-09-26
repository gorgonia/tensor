package tensor

import (
	"github.com/chewxy/inigo/values/tensor/internal"
	"gorgonia.org/shapes"
)

type MemoryFlag = internal.MemoryFlag

type DataOrder = internal.DataOrder

type Triangle = internal.Triangle

type Shape = shapes.Shape

type Memory = internal.Memory

type Engine = internal.Engine

type Iterator = internal.Iterator

const (
	ColMajor      DataOrder = internal.ColMajor
	NonContiguous           = internal.NonContiguous
	Transposed              = internal.Transposed
)

type SliceRange = shapes.Slice

type Num = internal.Num

const NativelyInaccessible = internal.NativelyInaccessible
