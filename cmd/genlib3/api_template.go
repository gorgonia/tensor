package main

import "text/template"

const apiArithOpRaw = `// {{.Name}} performs ` + "`t {{.Symbol}} u`." + `
func {{.Name}}[DT Num](t, u Basic[DT], opts ...FuncOpt)(Basic[DT], error) {
	e, newAPT, newAPU, retVal, fo, err :=  PrepBasicBinOpCis[DT](t, u, opts...)
	if err != nil {
		return nil, err
	}

	ctx := fo.Ctx
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast.BroadcastData()

	{{.Interface | lower}}, ok := e.({{.Interface}}[DT, Basic[DT]]);
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, {{.Interface | lower}}, errors.ThisFn())
	}
	switch {
	case toBroadcast:
		err = {{.Interface|lower}}.{{.Name}}Broadcastable(ctx, t, u, retVal, newAPT, newAPU, toIncr)
	default:
		if err := checkCompatibleShape(t.Shape(), u.Shape())(); err != nil{
			return nil, err
		}
		if err = {{.Interface|lower}}.{{.Name}}(ctx, t, u, retVal, toIncr); err !=nil{
			return nil, err
		}
		err = postOpBroadcastReshape(fo.Broadcast, t, u, retVal)
	}
	return retVal, err
}
`

const apiCmpOpRaw = `// {{.Name}} performs ` + "`t {{.Symbol}} u`." + `
func {{.Name}}[DT Num](t, u Basic[DT], opts ...FuncOpt)(DescWithStorage, error) {
	e, newAPT, newAPU, retVal, fo, err :=  PrepBinOpTrans[DT](t, u, opts...)
	if err != nil {
		return nil, err
	}

	asSame := fo.AsType == t.Dtype()
	ctx := fo.Ctx
	toBroadcast := fo.Broadcast.BroadcastData()


	{{.Interface | lower}}, ok := e.({{.Interface}}[DT, Basic[DT]]);
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, {{.Interface | lower}}, errors.ThisFn())
	}
	if fo.Incr {
		return nil, errors.Errorf("Unable to perform Incr for {{.Name}}")
	}
	switch {
	case toBroadcast:
		err = {{.Interface|lower}}.{{.Name}}Broadcastable(ctx, t, u, retVal, asSame, newAPT, newAPU)
	default:
		if err := checkCompatibleShape(t.Shape(), u.Shape())(); err != nil{
			return nil, err
		}
		if err = {{.Interface|lower}}.{{.Name}}(ctx, t, u, retVal, asSame); err !=nil{
			return nil, err
		}
		err = postOpBroadcastReshape(fo.Broadcast, t, u, retVal)
	}
	return retVal, err
}
`

var (
	apiArithOp *template.Template
	apiCmpOp   *template.Template
)

func init() {
	apiArithOp = template.Must(template.New("APIArith").Funcs(funcs).Parse(apiArithOpRaw))
	apiCmpOp = template.Must(template.New("APICmp").Funcs(funcs).Parse(apiCmpOpRaw))
}
