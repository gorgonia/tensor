package stdeng_test

import (
	"testing"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
	. "gorgonia.org/tensor/engines"
	"gorgonia.org/tensor/internal/errors"
	gutils "gorgonia.org/tensor/internal/utils"
)

var _ tensor.FuncOptHandler[int] = StdOrderedNumEngine[int, *dense.Dense[int]]{}

// CompatAdd is a compatibility function
func CompatAdd[DT any](a, b tensor.Basic[DT], opts ...tensor.FuncOpt) (retVal tensor.Basic[DT], err error) {
	e := a.Engine()

	var prepper tensor.FuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(tensor.FuncOptHandler[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}
	var fo tensor.Option
	if retVal, fo, err = prepper.HandleFuncOpts(a, a.Shape(), opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	toIncr := fo.Incr
	ctx := fo.Ctx

	adder, ok := e.(tensor.Adder[DT])
	if !ok {
		return nil, errors.Errorf("Expected engine to be an Adder")
	}
	if err = adder.Add(ctx, a, b, retVal, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}

func Test_CompatAdd(t *testing.T) {

	var a, b *dense.Dense[int]
	a = dense.New[int](WithShape(2, 2), WithBacking(gutils.Range[int](0, 4)))
	b = dense.New[int](WithShape(3, 3), WithBacking(gutils.Range[int](0, 9)))
	ret, err := CompatAdd[int](a, b)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", ret)
}
