package main

import "text/template"

const enginesCmpBinOpRaw = `// {{.Name | lower}}Op creates a ` + "`CmpBinOp`" + ` for values of the ` + "`{{.TypeClass}}`" + ` datatype.
func {{.Name | lower}}Op[DT {{.TypeClass}}]() CmpBinOp[DT] {
	return CmpBinOp[DT]{
		VV: execution.{{.Name | title}}VVBool[DT],
		VVBC: execution.{{.Name | title}}BCBool[DT],
		VVIter: execution.{{.Name | title}}VVIterBool[DT],

		VS: execution.{{.Name | title}}VSBool[DT],
		VSIter: execution.{{.Name | title}}VSIterBool[DT],

		SV: execution.{{.Name | title}}SVBool[DT],
		SVIter: execution.{{.Name | title}}SVIterBool[DT],
	}
}
`

const enginesOrderedNumOpRaw = `// {{.Name | lower}}OpOrderedNum creates the ops necessary for an OrderedNum engine.
func {{.Name | lower}}OpOrderedNum[DT {{.TypeClass}}]() (Op[DT], CmpBinOp[DT]){
	return Op[DT]{
		VV: execution.{{.Name | title}}VV[DT],
		VVBC: execution.{{.Name | title}}BC[DT],
		VVIter: execution.{{.Name | title}}VVIter[DT],

		VS: execution.{{.Name | title}}VS[DT],
		VSIter: execution.{{.Name | title}}VSIter[DT],

		SV: execution.{{.Name | title}}SV[DT],
		SVIter: execution.{{.Name | title}}SVIter[DT],
	}, {{.Name | lower}}Op[DT]()
}
`

const compComparableEngMethodsRaw = `// {{.Name}} performs ` + "`a {{.Symbol}} b`" + `, with a bool tensor as the return value. If` + "`asSameDT == true`" + `, an error will be returned.
func (e compComparableEng[DT, T]) {{.Name}}(ctx context.Context, a, b tensor.Basic[DT], retVal DescWithStorage, asSameDT bool) (err error) {
	op := {{.Name | lower}}Op[DT]()
	if asSameDT {
		var v DT
		return errors.Errorf("Unable to perform %v for data type %T", errors.ThisFn(), v)
	}
	return e.CmpOp(ctx, a, b, retVal.(tensor.Basic[bool]), op)
}

// {{.Name}} performs` + " `vec {{.Symbol}} scalar` or `scalar {{.Symbol}} vec`" + `, with a bool tensor as the return value. The ` + "`scalarOnLeft`" + ` parameter indicates
// if the scalar value is on the left of the bin op. If ` + "`asSameDT` == true" + `, an error will be returned.
func (e compComparableEng[DT, T]) {{.Name}}Scalar(ctx context.Context, a tensor.Basic[DT], b DT, retVal DescWithStorage, scalarOnLeft bool, asSameDT bool) (err error) {
	op := eleqOp[DT]()
	if asSameDT {
		var v DT
		return errors.Errorf("Unable to perform %v for data type %T", errors.ThisFn(), v)
	}
	return e.CmpOpScalar(ctx, a, b, retVal.(tensor.Basic[bool]), scalarOnLeft, op)
}

// {{.Name}}Broadcastable performs ` + "`a {{.Symbol}} b`" + `, with a bool tensor as the return value. The operation is broadacasted correctly according to shape. If` + "`asSameDT == true`" + `, an error will be returned.
func (e compComparableEng[DT, T]) {{.Name}}Broadcastable(ctx context.Context, a, b tensor.Basic[DT], retVal DescWithStorage, asSameDT bool, expAPA, expAPB *tensor.AP) (err error) {
	op := {{.Name | lower}}Op[DT]()
	if asSameDT {
		var v DT
		return errors.Errorf("Unable to perform %v for data type %T", errors.ThisFn(), v)
	}
	return e.CmpOpBC(ctx, a, b, retVal.(tensor.Basic[bool]),expAPA, expAPB, op)
}
`

const orderedEngMethodsRaw = `// {{.Name}} performs ` + "`a {{.Symbol}} b`" + `, with a bool tensor as the return value. If` + "`asSameDT == true`" + `, an error will be returned.
func (e OrderedEng[DT, T]) {{.Name}}(ctx context.Context, a, b tensor.Basic[DT], retVal DescWithStorage, asSameDT bool) (err error) {
	op := {{.Name | lower}}Op[DT]()
	if asSameDT {
		var v DT
		return errors.Errorf("Unable to perform %v for data type %T", errors.ThisFn(), v)
	}
	return e.CmpOp(ctx, a, b, retVal.(tensor.Basic[bool]), op)
}

// {{.Name}}Scalar performs` + " `vec {{.Symbol}} scalar` or `scalar {{.Symbol}} vec`" + `, with a bool tensor as the return value. The ` + "`scalarOnLeft`" + ` parameter indicates
// if the scalar value is on the left of the bin op. If ` + "`asSameDT` == true" + `, an error will be returned.
func (e OrderedEng[DT, T]) {{.Name}}Scalar(ctx context.Context, a tensor.Basic[DT], b DT, retVal DescWithStorage, scalarOnLeft bool, asSameDT bool) (err error) {
	op := {{.Name | lower}}Op[DT]()
	if asSameDT {
		var v DT
		return errors.Errorf("Unable to perform %v for data type %T", errors.ThisFn(), v)
	}
	return e.CmpOpScalar(ctx, a, b, retVal.(tensor.Basic[bool]), scalarOnLeft, op)
}


// {{.Name}}Broadcastable performs ` + "`a {{.Symbol}} b`" + `, with a bool tensor as the return value. The operation is broadacasted correctly according to shape. If` + "`asSameDT == true`" + `, an error will be returned.
func (e OrderedEng[DT, T]) {{.Name}}Broadcastable(ctx context.Context, a, b tensor.Basic[DT], retVal DescWithStorage, asSameDT bool, expAPA, expAPB *tensor.AP) (err error) {
	op := {{.Name | lower}}Op[DT]()
	if asSameDT {
		var v DT
		return errors.Errorf("Unable to perform %v for data type %T", errors.ThisFn(), v)
	}
	return e.CmpOpBC(ctx, a, b, retVal.(tensor.Basic[bool]),expAPA, expAPB, op)
}
`

const orderedNumEngMethodsRaw = `// {{.Name}} performs ` + "`a {{.Symbol}} b`" + `, with a bool tensor as the return value.
func (e StdOrderedNumEngine[DT, T]) {{.Name}}(ctx context.Context, a, b tensor.Basic[DT], retVal DescWithStorage, asSameDT bool) (err error) {
	op, cmpOp := {{.Name | lower}}OpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOp(ctx, a, b, retVal.(tensor.Basic[bool]), cmpOp)
	}
	return e.cmpOp(ctx, a, b, retVal.(tensor.Basic[DT]), op)
}

// {{.Name}}Scalar performs` + " `vec {{.Symbol}} scalar` or `scalar {{.Symbol}} vec`" + `, with a bool tensor as the return value. The ` + "`scalarOnLeft`" + ` parameter indicates
// if the scalar value is on the left of the bin op
func (e StdOrderedNumEngine[DT, T]) {{.Name}}Scalar(ctx context.Context, a tensor.Basic[DT], b DT, retVal DescWithStorage, scalarOnLeft bool, asSameDT bool) (err error) {
	op, cmpOp := {{.Name | lower}}OpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOpScalar(ctx, a, b, retVal.(tensor.Basic[bool]), scalarOnLeft, cmpOp)
	}
	return e.cmpOpScalar(ctx, a, b, retVal.(tensor.Basic[DT]), scalarOnLeft, op)
}


// {{.Name}}Broadcastable performs ` + "`a {{.Symbol}} b`" + `, with a bool tensor as the return value. The operation is broadacasted correctly according to shape. If` + "`asSameDT == true`" + `, an error will be returned.
func (e StdOrderedNumEngine[DT, T]) {{.Name}}Broadcastable(ctx context.Context, a, b tensor.Basic[DT], retVal DescWithStorage, asSameDT bool, expAPA, expAPB *tensor.AP) (err error) {
	op, cmpOp := {{.Name | lower}}OpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOpBC(ctx, a, b, retVal.(tensor.Basic[bool]), expAPA, expAPB, cmpOp )
	}
	return e.cmpOpBC(ctx, a, b, retVal.(tensor.Basic[DT]), expAPA, expAPB, op)
}
`

type engineOp struct {
	BinOp
}

var (
	enginesCmpBinOp     *template.Template
	enginesOrderedNumOp *template.Template

	compComparableEngMethods *template.Template
	orderedEngMethods        *template.Template
	orderedNumEngMethods     *template.Template
)

func init() {
	enginesCmpBinOp = template.Must(template.New("enginesCmpBinOp").Funcs(funcs).Parse(enginesCmpBinOpRaw))
	enginesOrderedNumOp = template.Must(template.New("enginesOp").Funcs(funcs).Parse(enginesOrderedNumOpRaw))

	compComparableEngMethods = template.Must(template.New("compComparableEngMethod").Funcs(funcs).Parse(compComparableEngMethodsRaw))
	orderedEngMethods = template.Must(template.New("orderedEngMethods").Funcs(funcs).Parse(orderedEngMethodsRaw))
	orderedNumEngMethods = template.Must(template.New("orderedNumEngMethods").Funcs(funcs).Parse(orderedNumEngMethodsRaw))
}
