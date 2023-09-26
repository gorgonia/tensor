package axialiter

import (
	"github.com/chewxy/inigo/values/tensor"
	"github.com/chewxy/inigo/values/tensor/internal"
	"gorgonia.org/shapes"
)

type AP = tensor.AP
type SliceRange = tensor.SliceRange
type Shape = shapes.Shape

func SR(start int, opts ...int) SliceRange { return internal.SR(start, opts...) }

var (
	Ltoi = tensor.Ltoi
)
