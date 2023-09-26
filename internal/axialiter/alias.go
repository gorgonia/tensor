package axialiter

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/shapes"
)

type AP = tensor.AP
type SliceRange = tensor.SliceRange
type Shape = shapes.Shape

func SR(start int, opts ...int) SliceRange { return internal.SR(start, opts...) }

var (
	Ltoi = tensor.Ltoi
)
