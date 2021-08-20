package tensor

import "github.com/pkg/errors"

func (t *Dense) Sum(along ...int) (retVal *Dense, err error) {
	e := t.Engine()
	ctx := ctxFromEngine(e)
	if sumer, ok := e.(Sumer); ok {
		var ret Tensor
		if ret, err = sumer.Sum(ctx, t, along...); err != nil {
			return
		}
		if retVal, err = assertDense(ret); err != nil {
			return nil, errors.Wrapf(err, opFail, "Sum")
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Sum")
}

func (t *Dense) Prod(along ...int) (retVal *Dense, err error) {
	e := t.Engine()
	ctx := ctxFromEngine(e)
	if sumer, ok := e.(Proder); ok {
		var ret Tensor
		if ret, err = sumer.Prod(ctx, t, along...); err != nil {
			return
		}
		if retVal, err = assertDense(ret); err != nil {
			return nil, errors.Wrapf(err, opFail, "Prod")
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Prod")
}

func (t *Dense) Max(along ...int) (retVal *Dense, err error) {
	e := t.Engine()
	ctx := ctxFromEngine(e)
	if maxer, ok := e.(Maxer); ok {
		var ret Tensor
		if ret, err = maxer.Max(ctx, t, along...); err != nil {
			return
		}
		if retVal, err = assertDense(ret); err != nil {
			return nil, errors.Wrapf(err, opFail, "Max")
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Max")
}

func (t *Dense) Min(along ...int) (retVal *Dense, err error) {
	e := t.Engine()
	ctx := ctxFromEngine(e)
	if miner, ok := e.(Miner); ok {
		var ret Tensor
		if ret, err = miner.Min(ctx, t, along...); err != nil {
			return
		}
		if retVal, err = assertDense(ret); err != nil {
			return nil, errors.Wrapf(err, opFail, "Min")
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Min")
}
