package tensor

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

func check(fns ...checkfunc) error {
	for _, fn := range fns {
		if err := fn(); err != nil {
			return err
		}
	}
	return nil
}
