package execution

import (
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

type nooperror interface {
	NoOp()
}

func handleNoOp(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := err.(nooperror); ok {
		return nil
	}
	return err
}

func iterCheck3[T, U, V any](a []T, b []U, c []V, ait, bit, cit Iterator) (cisa, cisb bool, err error) {
	cisa = cit == ait
	cisb = cit == bit

	// check: if cit == ait, then c == a
	// and likewise for cit == bit
	switch {
	case cisa:
		if !internal.SliceEqMeta(c, a) {
			return cisa, cisb, errors.Errorf("If citer == aiter, then the backing slices c and a must be the same")
		}
	case cisb:
		if !internal.SliceEqMeta(c, b) {
			return cisa, cisb, errors.Errorf("If citer == aiter, then the backing slices c and b must be the same")
		}
	}
	return cisa, cisb, nil
}

func iterCheck2[T, U any](a []T, b []U, ait, bit Iterator) (aisb bool, err error) {
	aisb = ait == bit

	// check: if ait == bit, then a == b

	if aisb && !internal.SliceEqMeta(a, b) {
		return aisb, errors.Errorf("If aiter == biter, then the backing slices a and b must be the same")
	}

	return aisb, nil
}
