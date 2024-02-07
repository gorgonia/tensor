package dense

import (
	"context"
	"fmt"
	"testing"

	"gorgonia.org/tensor"
	stdeng "gorgonia.org/tensor/engines"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/execution"
	gutils "gorgonia.org/tensor/internal/utils"

	"gorgonia.org/tensor/internal/specialized"
)

// Benchmark against a non specialized function

type specializedAdder[DT any, T tensor.Basic[DT]] interface {
	AddSpecialized(ctx context.Context, a, b, retVal T, toIncr bool) error
}

type interfaceAdder[DT any] interface {
	AddInterface(ctx context.Context, a, b, retVal tensor.Basic[DT], toIncr bool) error
}

func (e StdFloat64Engine[T]) AddInterface(ctx context.Context, a, b, retVal tensor.Basic[float64], toIncr bool) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return
	}

	var ait, bit, iit Iterator
	var useIter bool
	if ait, bit, iit, useIter, _, err = stdeng.PrepDataVV[float64](a, b, retVal); err != nil {
		return errors.Wrap(err, "Unable to prepare iterators for Add")
	}

	if useIter {
		switch {
		case toIncr:
			execution.AddVVIncrIter(a.Data(), b.Data(), retVal.Data(), ait, bit, iit)
			return nil
		default:
			execution.AddVVIter(a.Data(), b.Data(), retVal.Data(), ait, bit, iit)
			return nil
		}
	}

	switch {
	case toIncr:
		execution.AddVVIncr(a.Data(), b.Data(), retVal.Data())
		return nil
	default:
		// copy(retVal.Data(), a.Data())
		//vecf64.AddRecv(a.Data(), b.Data(), retVal.Data())
		execution.AddVV(a.Data(), b.Data(), retVal.Data())
		return nil
	}
}

func (e StdFloat64Engine[T]) AddSpecialized(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return
	}

	var ait, bit, iit Iterator
	var useIter bool
	if ait, bit, iit, useIter, _, err = stdeng.PrepDataVV[float64, float64](a, b, retVal); err != nil {
		return errors.Wrap(err, "Unable to prepare iterators for Add")
	}

	if useIter {
		switch {
		case toIncr:
			execution.AddVVIncrIter(a.Data(), b.Data(), retVal.Data(), ait, bit, iit)
			return nil
		default:
			execution.AddVVIter(a.Data(), b.Data(), retVal.Data(), ait, bit, iit)
			return nil
		}
	}

	switch {
	case toIncr:
		execution.AddVVIncr(a.Data(), b.Data(), retVal.Data())
		return nil
	default:
		// copy(retVal.Data(), a.Data())
		//vecf64.AddRecv(a.Data(), b.Data(), retVal.Data())
		execution.AddVV(a.Data(), b.Data(), retVal.Data())
		return nil
	}
}

func (e StdFloat64Engine[T]) AddScalarSpecialized(ctx context.Context, a T, b float64, toIncr bool, retVal T) (err error) {
	panic("NYI")
}

func (t *Dense[DT]) Add_INTERFACE(u *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	e := tensor.GetEngine(t, u)
	if err = check(checkFlags(e, t, u), checkEqShape(t.Shape(), u.Shape())); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	var prepper tensor.FuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(tensor.FuncOptHandler[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}
	var fo Option
	var retValB tensor.Basic[DT]
	if retValB, fo, err = prepper.HandleFuncOpts(t, t.Shape(), opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}

	toIncr := fo.Incr
	ctx := fo.Ctx

	var adder interfaceAdder[DT]
	if adder, ok = e.(interfaceAdder[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, adder, errors.ThisFn())
	}

	if err = adder.AddInterface(ctx, t, u, retValB, toIncr); err != nil {
		return nil, err
	}
	return retValB.(*Dense[DT]), nil
}

func (t *Dense[DT]) Add_SPECIALIZED(u *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	e := tensor.GetEngine(t, u)
	if err = check(checkFlags(e, t, u), checkEqShape(t.Shape(), u.Shape())); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	var prepper specialized.FuncOptHandler[DT, *Dense[DT]]
	var ok bool
	if prepper, ok = e.(specialized.FuncOptHandler[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}
	var fo Option
	if retVal, fo, err = prepper.HandleFuncOptsSpecialized(t, t.Shape(), opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}

	toIncr := fo.Incr
	ctx := fo.Ctx

	var adder specialized.Adder[DT, *Dense[DT]]
	if adder, ok = e.(specialized.Adder[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, adder, errors.ThisFn())
	}

	if err = adder.AddSpecialized(ctx, t, u, retVal, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}

var matrixCases = []struct{ r, c int }{
	{1024, 512},
	{512, 512},
	{256, 64},
	{32, 32},
	{8, 512},
	{512, 8},
	{9, 513},
	{7, 511},
}

func BenchmarkAdd(b *testing.B) {

	for _, c := range matrixCases {
		b.Run(fmt.Sprintf("algorithm=specialized/size=(%d,%d)", c.r, c.c), func(b *testing.B) {
			b.StopTimer()
			T := New[float64](WithShape(c.r, c.c), WithBacking(gutils.Random[float64](c.r*c.c)))
			U := New[float64](WithShape(c.r, c.c), WithBacking(gutils.Random[float64](c.r*c.c)))
			var V *Dense[float64]
			var err error

			b.ResetTimer()
			b.StartTimer()
			for i := 0; i < b.N; i++ {
				V, err = T.Add_SPECIALIZED(U)
			}
			_ = V
			_ = err
		})

		b.Run(fmt.Sprintf("algorithm=standard/size=(%d,%d))", c.r, c.c), func(b *testing.B) {
			b.StopTimer()
			T := New[float64](WithShape(c.r, c.c), WithBacking(gutils.Random[float64](c.r*c.c)))
			U := New[float64](WithShape(c.r, c.c), WithBacking(gutils.Random[float64](c.r*c.c)))
			var V *Dense[float64]
			var err error

			b.ResetTimer()
			b.StartTimer()
			for i := 0; i < b.N; i++ {
				V, err = T.Add_INTERFACE(U)
			}
			_ = V
			_ = err
		})
	}

}
