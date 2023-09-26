package dense

import (
	"context"
	"fmt"
	"math"
	"testing"

	"gorgonia.org/tensor"
	stdeng "gorgonia.org/tensor/engines"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/specialized"
	gutils "gorgonia.org/tensor/internal/utils"
)

// benchmark `Apply` vs specialized unary function

type SqrterBENCH[DT any, T tensor.Tensor[DT, T]] interface {
	Sqrt_BENCH(ctx context.Context, a, retVal T) error
}

func sqrtF64s(data, retVal []float64) {
	for i := range data {
		retVal[i] = math.Sqrt(data[i])
	}
}

func sqrtF64sWithIterator(data, retVal []float64, ait, rit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = internal.HandleNoOp(err)
			break
		}
		if j, validj, err = rit.NextValidity(); err != nil {
			err = internal.HandleNoOp(err)
			break
		}
		if validi && validj {
			retVal[j] = math.Sqrt(data[i])
		}
	}
	return err
}

func (e StdFloat64Engine[T]) Sqrt_BENCH(ctx context.Context, a, retVal T) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return
	}

	var ait, rit Iterator
	var useIter bool
	if ait, rit, useIter, err = stdeng.PrepDataUnary[float64](a, retVal); err != nil {
		return errors.Wrap(err, "Unable to prepare iterators ofr Sqrt")
	}

	if useIter {
		// Not gonna be called during the benchmarks
		// slow case anyway
		return sqrtF64sWithIterator(a.Data(), retVal.Data(), ait, rit)
	}
	sqrtF64s(a.Data(), retVal.Data())
	return nil
}

func (t *Dense[DT]) Sqrt_Eng(opts ...FuncOpt) (retVal *Dense[DT], err error) {
	e := t.e
	if err = check(checkFlags(e, t)); err != nil {
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

	ctx := fo.Ctx
	var sqrter SqrterBENCH[DT, *Dense[DT]]
	if sqrter, ok = e.(SqrterBENCH[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, sqrter, errors.ThisFn())
	}
	if err = sqrter.Sqrt_BENCH(ctx, t, retVal); err != nil {
		return nil, err
	}
	return retVal, nil
}

func BenchmarkSqrt(b *testing.B) {
	for _, c := range matrixCases {
		b.Run(fmt.Sprintf("algorithm=specializedUnary/size=(%d,%d)", c.r, c.c), func(b *testing.B) {
			b.StopTimer()
			T := New[float64](WithShape(c.r, c.c), WithBacking(gutils.Random[float64](c.r*c.c)))
			U := New[float64](WithShape(c.r, c.c))
			var V *Dense[float64]
			var err error

			b.ResetTimer()
			b.StartTimer()
			for i := 0; i < b.N; i++ {
				V, err = T.Sqrt_Eng(WithReuse(U))
			}
			if U != V {
				b.Error("X")
			}
			_ = U
			_ = err
		})

		b.Run(fmt.Sprintf("algorithm=apply/size=(%d,%d)", c.r, c.c), func(b *testing.B) {
			b.StopTimer()
			T := New[float64](WithShape(c.r, c.c), WithBacking(gutils.Random[float64](c.r*c.c)))
			U := New[float64](WithShape(c.r, c.c))
			var V *Dense[float64]
			var err error
			fn := func(x float64) (float64, error) { return math.Sqrt(x), nil }

			b.ResetTimer()
			b.StartTimer()
			for i := 0; i < b.N; i++ {
				V, err = T.Apply(fn, WithReuse(U))
			}
			if U != V {
				b.Error("X")
			}
			_ = U
			_ = err
		})
	}
}
