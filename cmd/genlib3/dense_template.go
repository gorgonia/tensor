package main

import "text/template"

const basicArithPrep = `func (t *Dense[DT]) basicArithPrep(u *Dense[DT], opts ...FuncOpt) (e Engine, newAPT, newAPU *tensor.AP, retVal *Dense[DT], fo Option, err error) {
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

	newAPT = t.Info()
	newAPU = u.Info()
	if fo.Broadcast {
		// create autobroadcast shape
		newAPT, newAPU = tensor.CalcBroadcastShapes(t.Info(), u.Info())
		if err = tensor.CheckBroadcastable(newAPT.Shape(), newAPU.Shape()); err != nil {
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

const denseArithOpRaw = `// {{.Name}} performs ` + "`t {{.Symbol}} u`." + `
func (t *Dense[DT]) {{.Name}}(u *Dense[DT], opts ...FuncOpt)(*Dense[DT], error) {
	e, newAPT, newAPU, retVal, fo, err := t.basicArithPrep(u, opts...)
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

	switch {
	case toBroadcast:
		err = {{.Name|lower}}er.{{.Name}}Broadcastable(ctx, t, u, retVal, newAPT, newAPU, toIncr)
	default:
		err = {{.Name|lower}}er.{{.Name}}(ctx, t, u, retVal, toIncr)

	}
	return retVal, err
}

// {{.Name}}Scalar performs ` + "`t {{.Symbol}} s`. If `scalarOnLeft` is true, then it performs `s {{.Symbol}} t`." + `
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

const denseCmpOpRaw = `// {{.Name}} performs ` + "`t {{.Symbol}} u`" + `
func (t *Dense[DT]) {{.Name}}(u *Dense[DT], opts ...FuncOpt) (retVal DescWithStorage, err error) {
	e := getEngine[DT](t, u)
	if err = check(checkFlags(e, t, u)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	tShp := t.Shape()
	uShp := u.Shape()
	expShape := largestShape(tShp, uShp)

	var prepper tensor.DescFuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(tensor.DescFuncOptHandler[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}

	var fo Option
	if retVal, fo, err = prepper.HandleFuncOptsDesc(t, expShape, opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if fo.Incr {
		return nil, errors.Errorf("Unable to Incr for Lt")
	}

	asBool := fo.AsType == dtype.Bool
	ctx := fo.Ctx

	var cmper tensor.{{.Interface}}[DT, *Dense[DT]]
	if cmper, ok = e.(tensor.{{.Interface}}[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, cmper, errors.ThisFn())
	}

	if fo.Broadcast {
		// create Autobroadcast shape
		newAPT, newAPU := tensor.CalcBroadcastShapes(t.Info(), u.Info())
		if err = tensor.CheckBroadcastable(newAPT.Shape(), newAPU.Shape()); err != nil {
			return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
		}

		err = cmper.{{.Name}}Broadcastable(ctx, t, u, retVal, !asBool, newAPT, newAPU)
		return
	}

	if err = cmper.{{.Name}}(ctx, t, u, retVal, !asBool); err != nil {
		return nil, err
	}
	return retVal, nil
}
`

var (
	denseArithOp *template.Template
	denseCmpOp   *template.Template
)

func init() {
	denseArithOp = template.Must(template.New("denseArithOp").Funcs(funcs).Parse(denseArithOpRaw))
	denseCmpOp = template.Must(template.New("denseCmpOp").Funcs(funcs).Parse(denseCmpOpRaw))
}