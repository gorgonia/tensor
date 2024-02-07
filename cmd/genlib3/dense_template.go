package main

import "text/template"

const basicArithPrep = `func (t *Dense[DT]) basicArithPrep(u *Dense[DT], opts ...FuncOpt) (e Engine, newAPT, newAPU *tensor.AP, retVal *Dense[DT], fo Option, err error) {
	return tensor.PrepBinOpCis[DT,*Dense[DT]](t, u, opts...)
}


func (t *Dense[DT]) basicArithScalarPrep(s DT, opts ...FuncOpt) (e Engine, retVal *Dense[DT], ctx context.Context, toIncr bool, err error) {
	e = tensor.GetEngine(t)
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
		if err := checkCompatibleShape(t.Shape(), u.Shape())(); err != nil{
			return retVal, err
		}
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

const basicCmpPrep = `func (t *Dense[DT]) basicCmpPrep(u *Dense[DT], opts ...FuncOpt) (e Engine, newAPT, newAPU *tensor.AP, retVal DescWithStorage, fo Option, err error) {
	return tensor.PrepBinOpTrans[DT](t, u, opts...)
}
`

const denseCmpOpRaw = `// {{.Name}} performs ` + "`t {{.Symbol}} u`" + `
func (t *Dense[DT]) {{.Name}}(u *Dense[DT], opts ...FuncOpt) (retVal DescWithStorage, err error) {
	e, newAPT, newAPU, retVal, fo, err := t.basicCmpPrep(u, opts...)
 	if err != nil {
		return nil, err
	}
	asSame := fo.AsType == t.Dtype()
	ctx := fo.Ctx
	toBroadcast := fo.Broadcast

	cmper, ok := e.(tensor.{{.Interface}}[DT, *Dense[DT]]);
 	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, cmper, errors.ThisFn())
	}
	if fo.Incr {
		return nil, errors.Errorf("Unable to Incr for {{.Name}}")
	}

	switch {
	case toBroadcast:
		err = cmper.{{.Name}}Broadcastable(ctx, t, u, retVal, asSame, newAPT, newAPU)
	default:
		if err := checkCompatibleShape(t.Shape(), u.Shape())(); err != nil{
			return retVal, err
		}
		err = cmper.{{.Name}}(ctx, t, u, retVal, asSame)

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
