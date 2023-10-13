package dense

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
)

func norm[DT float32 | float64, T tensor.Tensor[DT, T]](t T, ord tensor.NormOrder, axes []int, sqrt, abs, norm0, normN, ps func(DT) DT) (retVal tensor.Basic[DT], err error) {
	sq := func(a DT) DT { return a * a }

	switch len(axes) {
	case 1:
		switch {
		case ord.IsUnordered() || ord == tensor.Norm(2):
			ret, err := t.Apply(sq)
			if err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-2: square step")
			}
			r := any(ret).(*Dense[DT])
			if r, err = Sum(r, tensor.Along(axes...)); err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-2: sum step")
			}
			return r.Apply(sqrt, tensor.UseUnsafe)
		case ord.IsInf(1):
			ret, err := t.Apply(abs)
			if err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm +∞: abs")
			}
			return Max(any(ret).(*Dense[DT]), tensor.Along(axes...))
		case ord.IsInf(-1):
			ret, err := t.Apply(abs)
			if err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm -∞: abs")
			}
			return Min(any(ret).(*Dense[DT]), tensor.Along(axes...))
		case ord == tensor.Norm(0):
			ret, err := t.Apply(norm0)
			if err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-0: applying norm0")
			}
			return Sum(any(ret).(*Dense[DT]), tensor.Along(axes...))
		case ord == tensor.Norm(1):
			ret, err := t.Apply(abs)
			if err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-0: applying abs")
			}
			return Sum(any(ret).(*Dense[DT]), tensor.Along(axes...))
		default:
			ret, err := t.Apply(normN)
			if err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-0: applying normN")
			}
			r, err := Sum(any(ret).(*Dense[DT]), tensor.Along(axes...))
			if err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "NormN: sum step")
			}
			return r.Apply(ps, tensor.UseUnsafe)
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
