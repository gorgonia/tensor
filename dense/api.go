package dense

import (
	"golang.org/x/exp/constraints"
	"gorgonia.org/dtype"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/execution"
)

// Max returns the maximum value of the elements in the tensor. To return the maximum value of along one or more particular axes, use the Along funcopt.
func Max[DT constraints.Ordered](a *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	module := tensor.ReductionModule[DT]{
		MonotonicReduction: execution.MonotonicMax[DT],
		ReduceFirstN:       execution.Max0[DT],
		ReduceLastN:        execution.Max[DT],
		Reduce: func(a, b DT) DT {
			if a > b {
				return a
			}
			return b
		},
	}

	var z DT
	return a.Reduce(module, z, opts...)
}

// Min returns the minimum value of the elements in the tensor. To return the minimum value of along one or more particular axes, use the Along funcopt.
func Min[DT constraints.Ordered](a *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	module := tensor.ReductionModule[DT]{
		MonotonicReduction: execution.MonotonicMin[DT],
		ReduceFirstN:       execution.Min0[DT],
		ReduceLastN:        execution.Min[DT],
		Reduce: func(a, b DT) DT {
			if a < b {
				return a
			}
			return b
		},
	}

	var z DT
	return a.Reduce(module, z, opts...)
}

// Sum returns the sum of the elements in the tensor. To return the sum of along one or more particular axes, use the Along funcopt.
func Sum[DT Num](a *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	module := tensor.ReductionModule[DT]{
		MonotonicReduction: execution.MonotonicSum[DT],
		ReduceFirstN:       execution.Sum0[DT],
		ReduceLastN:        execution.Sum[DT],
		Reduce:             func(a, b DT) DT { return a + b },
	}

	var z DT
	return a.Reduce(module, z, opts...)
}

// Prod returns the product of the elements in the tensor. To return the product of along one or more particular axes, use the Along funcopt.
func Prod[DT Num](a *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	module := tensor.ReductionModule[DT]{
		MonotonicReduction: execution.MonotonicProd[DT],
		ReduceFirstN:       execution.Prod0[DT],
		ReduceLastN:        execution.Prod[DT],
		Reduce:             func(a, b DT) DT { return a * b },
	}

	var z DT = 1
	return a.Reduce(module, z, opts...)
}

func Lt[DT OrderedNum](a, b *Dense[DT], opts ...FuncOpt) (retVal DescWithStorage, err error) {
	e := getEngine[DT](a, b)
	if err = check(checkFlags(e, a, b), checkEqShape(a.Shape(), b.Shape())); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	var prepper tensor.DescFuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(tensor.DescFuncOptHandler[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}

	var fo Option
	if retVal, fo, err = prepper.HandleFuncOptsDesc(a, a.Shape(), opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if fo.Incr {
		return nil, errors.Errorf("Unable to Incr for Lt")
	}

	asBool := fo.AsType == dtype.Bool
	ctx := fo.Ctx

	var cmper tensor.Comparer[DT, *Dense[DT]]
	if cmper, ok = e.(tensor.Comparer[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, cmper, errors.ThisFn())
	}
	if err = cmper.Lt(ctx, a, b, retVal, !asBool); err != nil {
		return nil, err
	}
	return retVal, nil
}
