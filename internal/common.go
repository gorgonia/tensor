package internal

import (
	"context"
	"reflect"

	"golang.org/x/exp/constraints"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal/errors"
)

// AllAxes is a special integer representing all axes
const AllAxes = int(shapes.AllAxes)

// Range represents a slicing operation for a Tensor.
type Range struct {
	start, end, step int
}

func (r Range) Start() int { return r.start }
func (r Range) End() int   { return r.end }
func (r Range) Step() int  { return r.step }

// SR is a helper function to create a SliceRange.
func SR(start int, opts ...int) Range {
	end := start + 1
	step := 1
	if len(opts) > 0 {
		end = opts[0]
	}
	if len(opts) > 1 {
		step = opts[1]
	}
	return Range{start, end, step}
}

// SliceMem is an internal definition of Memory for a slice
type SliceMem struct {
	reflect.SliceHeader
	sz uintptr // size in bytes
}

func (m *SliceMem) Uintptr() uintptr { return m.Data }
func (m *SliceMem) MemSize() uintptr { return m.sz }

// IsHostMemory returns true because SliceMem will always be used for internal host memory only.
func (m *SliceMem) IsHostMemory() bool { return true }

// HandleNoOp handles noop errors - if NoOp is found, then nil is returned.
func HandleNoOp(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := err.(errors.NoOpError); ok {
		return nil
	}
	return err
}

// HandleCtx returns an error if the context had previously been cancelled.
func HandleCtx(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return errors.NoOp{}
	default:
	}
	return nil
}

// Min returns the minimum of two values.
func Min[DT constraints.Ordered](a, b DT) DT {
	if a < b {
		return a
	}
	return b
}

func Max[DT constraints.Ordered](a, b DT) DT {
	if a > b {
		return a
	}
	return b
}

func MakeMonotonicReduction[DT any](f func(a DT, b DT) DT, defaultValue DT) func(xs []DT) DT {
	return func(xs []DT) DT {
		var retVal DT = defaultValue
		for _, v := range xs {
			retVal = f(retVal, v)
		}
		return retVal
	}
}
