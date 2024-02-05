package dense

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal/errors"
)

type checkfunc func() error

func checkFlags(e Engine, ts ...DescWithStorage) checkfunc {
	return func() error {
		for _, t := range ts {
			if !e.WorksWith(t.Flags(), t.DataOrder()) {
				return errors.Errorf(errors.EngineIncompatibility, t.Flags(), t.DataOrder())
			}
		}
		return nil
	}
}

func checkEqShape(expected shapes.Shape, others ...shapes.Shape) checkfunc {
	return func() error {
		for _, s := range others {
			if !expected.Eq(s) {
				return errors.Errorf(errors.ShapeMismatch, expected, s)
			}
		}
		return nil
	}
}

func checkCompatibleShape(expected shapes.Shape, others ...shapes.Shape) checkfunc {
	return func() error {
		expLen := expected.TotalSize()
		for _, s := range others {
			if s.TotalSize() != expLen {
				return errors.Errorf(errors.ShapeMismatch, expected, s)
			}
		}
		return nil
	}
}

func checkDims(dims int, ts ...DescWithStorage) checkfunc {
	return func() error {
		for _, t := range ts {
			if t.Dims() != dims {
				return errors.Errorf(errors.DimMismatch, dims, t.Dims())
			}
		}
		return nil
	}
}

// checkInnerProdDims checks that the innermost dimension of `a` matches the outermost dimension of `b`
func checkInnerProdDims(a, b Desc) checkfunc {
	return func() error {
		aShape := a.Shape()
		bShape := b.Shape()
		if aShape[len(aShape)-1] != bShape[0] {
			return errors.Errorf(errors.ShapeMismatch2, aShape, bShape)
		}
		return nil
	}
}

func checkIsVector(a Desc) checkfunc {
	return func() error {
		if !a.Shape().IsVectorLike() {
			return errors.Errorf("Expected a vector. Got %v instead", a.Shape())
		}
		return nil
	}
}

func checkValidAxis(axis int, a Desc) checkfunc {
	return func() error {
		if axis >= a.Dims() {
			return errors.Errorf(errors.InvalidAxis, axis, a.Dims())
		}
		return nil
	}
}

func checkRepeatValidAxis(axis int, a Desc) checkfunc {
	return func() error {
		if a.Shape().IsScalar() {
			return nil
			// then  0 and 1 is ok
			// switch axis {
			// case 0, 1:
			// 	return nil
			// default:
			// 	return errors.Errorf(errors.InvalidAxis, axis, a.Dims())
			// }
		}
		if axis >= a.Dims() {
			return errors.Errorf(errors.InvalidAxis, axis, a.Dims())
		}
		return nil
	}
}

func check(fns ...checkfunc) error {
	for _, fn := range fns {
		if err := fn(); err != nil {
			return err
		}
	}
	return nil
}
