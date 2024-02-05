// Code generated by genlib3. DO NOT EDIT

package dense

import (
	"context"
	"math"

	"github.com/chewxy/math32"
)

func (e StdFloat64Engine[T]) Abs(ctx context.Context, a, retVal T) (err error) {
	fn := math.Abs
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Sign(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float64) float64 {
		if x < 0 {
			return -1
		}
		return 1
	}
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Ceil(ctx context.Context, a, retVal T) (err error) {
	fn := math.Ceil
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Floor(ctx context.Context, a, retVal T) (err error) {
	fn := math.Floor
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Neg(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float64) float64 { return -x }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Inv(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float64) float64 { return 1 / x }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) InvSqrt(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float64) float64 { return 1 / math.Sqrt(x) }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Exp(ctx context.Context, a, retVal T) (err error) {
	fn := math.Exp
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Log(ctx context.Context, a, retVal T) (err error) {
	fn := math.Log
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Log2(ctx context.Context, a, retVal T) (err error) {
	fn := math.Log2
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Log10(ctx context.Context, a, retVal T) (err error) {
	fn := math.Log10
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Log1p(ctx context.Context, a, retVal T) (err error) {
	fn := math.Log1p
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Expm1(ctx context.Context, a, retVal T) (err error) {
	fn := math.Expm1
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Square(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float64) float64 { return x * x }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Sqrt(ctx context.Context, a, retVal T) (err error) {
	fn := math.Sqrt
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Cube(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float64) float64 { return x * x * x }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat64Engine[T]) Tanh(ctx context.Context, a, retVal T) (err error) {
	fn := math.Tanh
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Abs(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Abs
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Sign(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float32) float32 {
		if x < 0 {
			return -1
		}
		return 1
	}
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Ceil(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Ceil
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Floor(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Floor
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Neg(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float32) float32 { return -x }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Inv(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float32) float32 { return 1 / x }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) InvSqrt(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float32) float32 { return 1 / math32.Sqrt(x) }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Exp(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Exp
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Log(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Log
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Log2(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Log2
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Log10(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Log10
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Log1p(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Log1p
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Expm1(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Expm1
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Square(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float32) float32 { return x * x }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Sqrt(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Sqrt
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Cube(ctx context.Context, a, retVal T) (err error) {
	fn := func(x float32) float32 { return x * x * x }
	return e.Map(ctx, fn, a, retVal)
}

func (e StdFloat32Engine[T]) Tanh(ctx context.Context, a, retVal T) (err error) {
	fn := math32.Tanh
	return e.Map(ctx, fn, a, retVal)
}
