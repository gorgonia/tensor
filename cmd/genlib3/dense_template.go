package main

import "text/template"

const basicArithPrep = `func (t *Dense[DT]) basicArithPrep(u *Dense[DT], opts ...FuncOpt) (e Engine, newShapeT, newShapeU shapes.Shape, retVal *Dense[DT], fo Option, err error) {
	e = getEngine[DT](t, u)
	if err = check(checkFlags(e, t, u)); err != nil {
		return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
	}
	tShp := t.Shape()
	uShp := u.Shape()
	expShape := largestShape(tShp, uShp)

	retVal, fo, err = handleFuncOpts[DT, *Dense[DT]](e, t, expShape, opts...)
	if err != nil {
		return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
	}

	newShapeT = tShp
	newShapeU = uShp
	if fo.Broadcast {
		// create autobroadcast shape
		newShapeT, newShapeU = tensor.CalcBroadcastShapes(newShapeT, newShapeU)
		if err = tensor.CheckBroadcastable(newShapeT, newShapeU); err != nil {
			return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
		}
	}

	return
}

func (t *Dense[DT]) basicArithScalarPrep(s DT, opts ...FuncOpt) (e Engine, retVal *Dense[DT], ctx context.Context, toIncr bool, err error) {
	e = getEngine[DT](t)
	if err = check(checkFlags(e, t)); err != nil {
		return nil, nil, nil, false, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
	}
	var fo Option
	retVal, fo, err = handleFuncOpts[DT, *Dense[DT]](e, t, t.Shape(), opts...)
	if err != nil {
		return nil, nil, nil, false, err
	}

	toIncr = fo.Incr
	ctx = fo.Ctx
	return
}

`

const denseArithOpRaw = `
func (t *Dense[DT]) {{.Name}}(u *Dense[DT], opts ...FuncOpt)(*Dense[DT], error) {
	e, newShapeU, newShapeT, retVal, fo, err := t.basicArithPrep(u, opts...)
	if err != nil {
		return nil, err
	}
	ctx := fo.Ctx
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast

	{{.Name|lower}}er, ok := e.(tensor.{{.Interface}}[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, {{.Name|lower}}er, errors.ThisFn())
	}
	_ = toBroadcast
	_ = newShapeU
	_ = newShapeT

	if err = {{.Name|lower}}er.{{.Name}}(ctx, t, u, retVal, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}

func (t *Dense[DT]) {{.Name}}Scalar(s DT, scalarOnLeft bool, opts ...FuncOpt) (*Dense[DT], error) {
	e, retVal, ctx, toIncr, err := t.basicArithScalarPrep(s, opts...)
	if err != nil {
		return nil, err
	}

	{{.Name|lower}}er, ok := e.(tensor.{{.Interface}}[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, {{.Name|lower}}er, errors.ThisFn())
	}

	if err = {{.Name|lower}}er.{{.Name}}Scalar(ctx, t, s, retVal, scalarOnLeft, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}
`

var (
	denseArithOp *template.Template
)

func init() {
	denseArithOp = template.Must(template.New("denseArithOp").Funcs(funcs).Parse(denseArithOpRaw))
}
