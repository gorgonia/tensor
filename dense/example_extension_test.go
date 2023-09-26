package dense_test

import (
	"context"

	"gorgonia.org/tensor/dense"
	stdeng "gorgonia.org/tensor/engines"
	"gorgonia.org/tensor/scalar"
)

type E[T dense.DenseTensor[Point, T]] struct {
	stdeng.StdEng[Point, T]
}

type Point struct {
	X, Y int
}

func (e E[T]) Add(ctx context.Context, a, b, c dense.Value[Point]) (dense.Value[Point], error) {
	switch a := any(a).(type) {
	case *dense.Dense[Point]:
		switch b := any(b).(type) {
		case *dense.Dense[Point]:
			adata := a.Data()
			bdata := b.Data()
			cdata := c.Data()
			for i := range adata {
				a := adata[i]
				b := bdata[i]
				c := Point{a.X + b.X, a.Y + b.Y}
				cdata[i] = c
			}
			return c, nil
		case scalar.Scalar[Point]:

		}
	case scalar.Scalar[Point]:
	default:
	}
	panic("NYI")
}
