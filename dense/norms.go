package dense

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
)

func multisvdnorm[DT float32 | float64](t *Dense[DT], rowAxis, colAxis int) (retVal *Dense[DT], err error) {
	if rowAxis > colAxis {
		rowAxis--
	}
	dims := t.Dims()

	if retVal, err = t.RollAxis(colAxis, dims, true); err != nil {
		return
	}

	if retVal, err = retVal.RollAxis(rowAxis, dims, true); err != nil {
		return
	}

	// manual, since SVD only works on matrices. In the future, this needs to be fixed when gonum's lapack works for float32
	// TODO: SVDFuture
	switch dims {
	case 2:
		retVal, _, _, err = retVal.SVD(false, false)
	case 3:
		toStack := make([]*Dense[DT], retVal.Shape()[0])
		for i := 0; i < retVal.Shape()[0]; i++ {
			// var sliced, ithS *Dense[DT]
			// if sliced, err = sliceDense(retVal, ss(i)); err != nil {
			// 	return
			// }

			// if ithS, _, _, err = sliced.SVD(false, false); err != nil {
			// 	return
			// }

			// toStack[i] = ithS
		}

		retVal, err = toStack[0].Stack(0, toStack[1:]...)
		return
	default:
		err = errors.Errorf("multiSVDNorm for dimensions greater than 3")
	}

	return
}

func norm[DT float32 | float64](t *Dense[DT], ord tensor.NormOrder, axes []int, sqrt, abs, norm0, normN, ps func(DT) DT) (retVal *Dense[DT], err error) {
	sq := func(a DT) DT { return a * a }

	switch len(axes) {
	case 1:
		switch {
		case ord.IsUnordered() || ord == tensor.Norm(2):
			if retVal, err = t.Apply(sq); err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-2: square step")
			}
			if retVal, err = Sum(retVal, tensor.Along(axes...)); err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-2: sum step")
			}
			return retVal.Apply(sqrt, tensor.UseUnsafe)
		case ord.IsInf(1):
			if retVal, err = t.Apply(abs); err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm +∞: abs")
			}
			return Max(retVal, tensor.Along(axes...))
		case ord.IsInf(-1):
			if retVal, err = t.Apply(abs); err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm -∞: abs")
			}
			return Min(retVal, tensor.Along(axes...))
		case ord == tensor.Norm(0):
			if retVal, err := t.Apply(norm0); err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-0: applying norm0")
			}
			return Sum(retVal, tensor.Along(axes...))
		case ord == tensor.Norm(1):
			if retVal, err := t.Apply(abs); err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-0: applying abs")
			}
			return Sum(retVal, tensor.Along(axes...))
		default:
			if retVal, err := t.Apply(normN); err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-0: applying normN")
			}
			if retVal, err = Sum(retVal, tensor.Along(axes...)); err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "NormN: sum step")
			}
			return retVal.Apply(ps, tensor.UseUnsafe)
		}
	case 2:
		rowAxis := axes[0]
		colAxis := axes[1]
		// checks
		if rowAxis < 0 {
			return nil, errors.Errorf("Row Axis %d is < 0", rowAxis)
		}
		if colAxis < 0 {
			return nil, errors.Errorf("Col Axis %d is < 0", colAxis)
		}

		if rowAxis == colAxis {
			return nil, errors.Errorf("Duplicate axes found. Row Axis: %d, Col Axis %d", rowAxis, colAxis)
		}

		switch {
		case ord == tensor.Norm(2):
			// SVD Norm
			// TODO
			return nil, errors.Errorf("MultiSVDNorm not yet implemented")
		case ord == tensor.Norm(-2):
			// SVD Norm
			// TODO
			return nil, errors.Errorf("MultiSVDNorm not yet implemented")
		case ord == tensor.Norm(1):
			if colAxis > rowAxis {
				colAxis--
			}
			ret, err := t.Apply(abs)
			if err != nil {
				return nil, errors.Wrapf(err, errors.OpFail, "Norm-1: applying abs")
			}
			r, err := Sum(any(ret).(*Dense[DT]), tensor.Along(rowAxis))
			if err != nil {
				return nil, errors.Wrapf(err, errors.OpFail, "Norm-1: sum step")
			}
			return Max(r, tensor.Along(colAxis))
		case ord == tensor.Norm(-1):
			if colAxis > rowAxis {
				colAxis--
			}
			ret, err := t.Apply(abs)
			if err != nil {
				return nil, errors.Wrapf(err, errors.OpFail, "Norm-1: applying abs")
			}
			r, err := Sum(any(ret).(*Dense[DT]), tensor.Along(rowAxis))
			if err != nil {
				return nil, errors.Wrapf(err, errors.OpFail, "Norm-1: sum step")
			}
			return Min(r, tensor.Along(colAxis))
		case ord == tensor.Norm(0):
			return nil, errors.Errorf("Norm of order 0 undefined for matrices")
		case ord.IsInf(1):
			if rowAxis > colAxis {
				rowAxis--
			}
			ret, err := t.Apply(abs)
			if err != nil {
				return nil, errors.Wrapf(err, errors.OpFail, "Norm +∞: abs")
			}
			r, err := Sum(any(ret).(*Dense[DT]), tensor.Along(rowAxis))
			if err != nil {
				return nil, errors.Wrapf(err, errors.OpFail, "Norm +∞: sum along row axis")
			}
			return Max(r, tensor.Along(colAxis), tensor.UseUnsafe)
		case ord.IsInf(-1):
			if rowAxis > colAxis {
				rowAxis--
			}
			ret, err := t.Apply(abs)
			if err != nil {
				return nil, errors.Wrapf(err, errors.OpFail, "Norm +∞: abs")
			}
			r, err := Sum(any(ret).(*Dense[DT]), tensor.Along(rowAxis))
			if err != nil {
				return nil, errors.Wrapf(err, errors.OpFail, "Norm +∞: sum along row axis")
			}
			return Min(r, tensor.Along(colAxis), tensor.UseUnsafe)
		case ord.IsUnordered() || ord.IsFrobenius():
			// TODO
		case ord.IsNuclear():
			// TODO

		}
	}
	panic("Unreachable")
}
