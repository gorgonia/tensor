package dense

import (
	"context"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/execution"
	"gorgonia.org/tensor/internal/specialized"
	"golang.org/x/exp/constraints"
	"gorgonia.org/dtype"
)

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

	var prepper specialized.FuncOptHandler[DT, *Dense[DT]]
	var ok bool
	if prepper, ok = e.(specialized.FuncOptHandler[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}

	var fo Option
	if retVal, fo, err = prepper.HandleFuncOptsSpecialized(a, a.Shape(), opts...); err != nil {
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
	if err = cmper.Lt(ctx, a, b, retVal, asBool); err != nil {
		return nil, err
	}
	return retVal, nil
}

func Scatter[DT any](a *Dense[DT], indices Densor[int]) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(a.Engine(), a)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	//TODO: support funcops
	indicesT := indices.GetDense()
	maxT, err := Max[int](indicesT)
	if err != nil {
		return nil, err
	}
	max := maxT.ScalarValue()

	expShape := indicesT.Shape().Clone()
	expShape[len(expShape)-1] = max + 1
	retVal = New[DT](WithShape(expShape...), WithEngine(a.Engine()))

	var sc tensor.Scatterer[DT, *Dense[DT]]
	var ok bool
	if sc, ok = a.e.(tensor.Scatterer[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, a.e, sc, errors.ThisFn())
	}
	err = sc.Scatter(context.Background(), a, indicesT, retVal)
	return
}

func SelectByIndices[DT any](a *Dense[DT], axis int, indices tensor.Basic[int], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(a.Engine(), a)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	panic("NYI")
}

func SelectByIndicesB[DT any](input *Dense[DT], axis int, outGrad *Dense[DT], indices *Dense[int], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	panic("NYI")
}
