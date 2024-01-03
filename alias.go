package tensor

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal"
)

type MemoryFlag = internal.MemoryFlag

type DataOrder = internal.DataOrder

type Triangle = internal.Triangle

type Shape = shapes.Shape

type Memory = internal.Memory

type Engine = internal.Engine

type Iterator = internal.Iterator

const (
	NativelyInaccessible MemoryFlag = internal.NativelyInaccessible
	ManuallyManaged                 = internal.ManuallyManaged
	IsOverallocated                 = internal.IsOverallocated
	IsView                          = internal.IsView
)

const (
	ColMajor      DataOrder = internal.ColMajor
	NonContiguous           = internal.NonContiguous
	Transposed              = internal.Transposed
)

type SliceRange = shapes.Slice

type Num = internal.Num

// Range creates a slice of values of the given DT
func Range[DT internal.Rangeable](start, end int) []DT {
	size := end - start
	incr := true
	if start > end {
		size = start - end
		incr = false
	}
	if size < 0 {
		panic("Cannot create a range that is negative in size")
	}
	retVal := make([]DT, size)
	for i, v := 0, DT(start); i < size; i++ {
		retVal[i] = v
		if incr {
			v++
		} else {
			v--
		}
	}
	return retVal
}

// MakeMemoryFlag makes a memory flag
// TODO use golinkname
func MakeMemoryFlag(fs ...MemoryFlag) (retVal MemoryFlag) { return internal.MakeMemoryFlag(fs...) }
