// package specialized contains the interfaces that could be in package tensor, but are put here to prevent pollution of names
package specialized

import (
	"context"

	"github.com/chewxy/inigo/values/tensor"
)

type Adder[DT any, T tensor.Tensor[DT, T]] interface {
	AddSpecialized(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	AddScalarSpecialized(ctx context.Context, a T, b DT, toIncr bool, retVal T) (err error)
}

type FuncOptHandler[DT any, T tensor.Tensor[DT, T]] interface {
	HandleFuncOptsSpecialized(a T, expShape tensor.Shape, opts ...tensor.FuncOpt) (retVal T, fo tensor.Option, err error)
}
