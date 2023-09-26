package stdeng_test

import (
	"context"
	"testing"

	"github.com/chewxy/inigo/values/tensor"
	"github.com/chewxy/inigo/values/tensor/dense"
	stdeng "github.com/chewxy/inigo/values/tensor/engines"
)

type BasicAdder[DT any] interface {
	Add(ctx context.Context, a, b, retVal tensor.Basic[DT], toIncr bool) (err error)
	AddScalar(ctx context.Context, t tensor.Basic[DT], s DT, retVal tensor.Basic[DT], scalarOnLeft, toIncr bool) (err error)
}

type SpecializedAdder[DT any, T tensor.Tensor[DT, T]] interface {
	Add(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	AddScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)
}

func TestTransform(t *testing.T) {
	e := stdeng.StdNumEngine[float64, *dense.Dense[float64]]{}

	_, ok := any(e).(SpecializedAdder[float64, *dense.Dense[float64]])
	if !ok {
		t.Fatal("Wat")
	}
	e2 := e.BasicEng()

	adder, ok := any(e2).(BasicAdder[float64])
	if !ok {
		t.Errorf("Expected BasicEng to make sure it implements specializedAdder")
	}
	_ = adder
}
