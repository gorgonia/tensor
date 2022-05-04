package main

import (
	"fmt"
	"io"
	"reflect"
	"strings"
	"text/template"
)

type InternalEngArithMethod struct {
	BinOp
	Kinds    []reflect.Kind
	Incr     bool
	Iter     bool
	WithRecv bool
}

type eLoopBody struct {
	BinOp

	Err   bool
	Kinds []reflect.Kind
}

func (fn *InternalEngArithMethod) Name() string {
	switch {
	case fn.Incr && fn.Iter:
		return fmt.Sprintf("%sIterIncr", fn.BinOp.Name())
	case fn.Incr && !fn.Iter:
		return fmt.Sprintf("%sIncr", fn.BinOp.Name())
	case !fn.Incr && fn.Iter:
		return fmt.Sprintf("%sIter", fn.BinOp.Name())
	case fn.WithRecv:
		return fmt.Sprintf("%sRecv", fn.BinOp.Name())
	default:
		return fn.BinOp.Name()
	}
}

func (fn *InternalEngArithMethod) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	switch {
	case fn.Iter && fn.Incr:
		paramNames = []string{"t", "a", "b", "incr", "ait", "bit", "iit"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, arrayType, iteratorType, iteratorType, iteratorType}
	case fn.Iter && !fn.Incr:
		paramNames = []string{"t", "a", "b", "ait", "bit"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, iteratorType, iteratorType}
	case !fn.Iter && fn.Incr:
		paramNames = []string{"t", "a", "b", "incr"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, arrayType}
	case fn.WithRecv:
		paramNames = []string{"t", "a", "b", "recv"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, arrayType}
	default:
		paramNames = []string{"t", "a", "b"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType}

	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Err:            true,
	}
}

func (fn *InternalEngArithMethod) WriteBody(w io.Writer) {
	var T *template.Template
	switch {
	case fn.Incr && fn.Iter:
		T = eArithIterIncr
	case fn.Incr && !fn.Iter:
		T = eArithIncr
	case fn.Iter && !fn.Incr:
		T = eArithIter
	case fn.WithRecv:
		T = eArithRecv
	default:
		T = eArith
	}
	lb := eLoopBody{
		BinOp: fn.BinOp,
		Kinds: fn.Kinds,
	}
	T.Execute(w, lb)
}

func (fn *InternalEngArithMethod) Write(w io.Writer) {
	w.Write([]byte("func (e E) "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateEArith(f io.Writer, kinds Kinds) {
	var methods []*InternalEngArithMethod
	for _, bo := range arithBinOps {
		var ks []reflect.Kind
		for _, k := range kinds.Kinds {
			if tc := bo.TypeClass(); tc != nil && tc(k) {
				ks = append(ks, k)
			}
		}
		meth := &InternalEngArithMethod{
			BinOp: bo,
			Kinds: ks,
		}
		methods = append(methods, meth)
	}

	// write vanilla
	for _, meth := range methods {
		meth.Write(f)
		meth.Incr = true
	}

	// write incr
	for _, meth := range methods {
		meth.Write(f)
		meth.Incr = false
		meth.Iter = true
	}

	// write iter
	for _, meth := range methods {
		meth.Write(f)
		meth.Incr = true
	}

	// write iter incr
	for _, meth := range methods {
		meth.Write(f)
		meth.Incr = false
		meth.Iter = false
	}

	// write recv
	for _, meth := range methods {
		meth.WithRecv = true
		meth.Write(f)
	}
}

/* MAP */

type InternalEngMap struct {
	Kinds []reflect.Kind
	Iter  bool
}

func (fn *InternalEngMap) Signature() *Signature {
	paramNames := []string{"t", "fn", "a", "incr"}
	paramTemplates := []*template.Template{reflectType, interfaceType, arrayType, boolType}
	name := "Map"
	if fn.Iter {
		paramNames = append(paramNames, "ait")
		paramTemplates = append(paramTemplates, iteratorType)
		name += "Iter"
	}

	return &Signature{
		Name:           name,
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Err:            true,
	}
}

func (fn *InternalEngMap) WriteBody(w io.Writer) {
	T := eMap
	if fn.Iter {
		T = eMapIter
	}
	template.Must(T.New("fntype0").Funcs(funcs).Parse(unaryFuncTypeRaw))
	template.Must(T.New("fntype1").Funcs(funcs).Parse(unaryFuncErrTypeRaw))

	lb := eLoopBody{
		Kinds: fn.Kinds,
	}
	T.Execute(w, lb)
}

func (fn *InternalEngMap) Write(w io.Writer) {
	w.Write([]byte("func (e E) "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)

	w.Write([]byte("\nreturn\n}\n\n"))
}

func generateEMap(f io.Writer, kinds Kinds) {
	m := new(InternalEngMap)
	for _, k := range kinds.Kinds {
		if isParameterized(k) {
			continue
		}
		m.Kinds = append(m.Kinds, k)
	}
	m.Write(f)
	m.Iter = true
	m.Write(f)
}

/* Cmp */

// InternalEngCmpMethod is exactly the same structure as the arith one, except it's Same instead of Incr.
// Some copy and paste leads to more clarity, rather than reusing the structure.
type InternalEngCmp struct {
	BinOp
	Kinds   []reflect.Kind
	RetSame bool
	Iter    bool
}

func (fn *InternalEngCmp) Name() string {
	switch {
	case fn.Iter && fn.RetSame:
		return fmt.Sprintf("%sSameIter", fn.BinOp.Name())
	case fn.Iter && !fn.RetSame:
		return fmt.Sprintf("%sIter", fn.BinOp.Name())
	case !fn.Iter && fn.RetSame:
		return fmt.Sprintf("%sSame", fn.BinOp.Name())
	default:
		return fn.BinOp.Name()
	}
}

func (fn *InternalEngCmp) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template

	switch {
	case fn.Iter && fn.RetSame:
		paramNames = []string{"t", "a", "b", "ait", "bit"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, iteratorType, iteratorType}
	case fn.Iter && !fn.RetSame:
		paramNames = []string{"t", "a", "b", "retVal", "ait", "bit", "rit"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, arrayType, iteratorType, iteratorType, iteratorType}
	case !fn.Iter && fn.RetSame:
		paramNames = []string{"t", "a", "b"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType}
	default:
		paramNames = []string{"t", "a", "b", "retVal"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, arrayType}
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Err: true,
	}
}

func (fn *InternalEngCmp) WriteBody(w io.Writer) {
	var T *template.Template
	switch {
	case fn.Iter && fn.RetSame:
		T = eCmpSameIter
	case fn.Iter && !fn.RetSame:
		T = eCmpBoolIter
	case !fn.Iter && fn.RetSame:
		T = eCmpSame
	default:
		T = eCmpBool
	}

	lb := eLoopBody{
		BinOp: fn.BinOp,
		Kinds: fn.Kinds,
	}
	T.Execute(w, lb)
}

func (fn *InternalEngCmp) Write(w io.Writer) {
	w.Write([]byte("func (e E) "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateECmp(f io.Writer, kinds Kinds) {
	var methods []*InternalEngCmp
	for _, bo := range cmpBinOps {
		var ks []reflect.Kind
		for _, k := range kinds.Kinds {
			if tc := bo.TypeClass(); tc != nil && tc(k) {
				ks = append(ks, k)
			}
		}
		meth := &InternalEngCmp{
			BinOp: bo,
			Kinds: ks,
		}
		methods = append(methods, meth)
	}

	for _, meth := range methods {
		meth.Write(f)
		meth.RetSame = true
	}

	for _, meth := range methods {
		meth.Write(f)
		meth.RetSame = false
		meth.Iter = true
	}
	for _, meth := range methods {
		meth.Write(f)
		meth.RetSame = true
	}

	for _, meth := range methods {
		meth.Write(f)
	}
}

/* MIN/MAX BETWEEN */

type InternalEngMinMaxBetween struct {
	BinOp
	Kinds []reflect.Kind
	Iter  bool
}

func (fn *InternalEngMinMaxBetween) Name() string {
	name := fn.BinOp.Name()

	switch {
	case fn.Iter:
		return fmt.Sprintf("%sBetweenIter", name)
	default:
		return name + "Between"
	}
}

func (fn *InternalEngMinMaxBetween) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template

	switch {
	case fn.Iter:
		paramNames = []string{"t", "a", "b", "ait", "bit"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, iteratorType, iteratorType}
	default:
		paramNames = []string{"t", "a", "b"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType}
	}
	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Err: true,
	}
}

func (fn *InternalEngMinMaxBetween) WriteBody(w io.Writer) {
	var T *template.Template

	switch {
	case fn.Iter:
		T = eMinMaxIter
	default:
		T = eMinMaxSame
	}

	lb := eLoopBody{
		BinOp: fn.BinOp,
		Kinds: fn.Kinds,
	}
	T.Execute(w, lb)
}

func (fn *InternalEngMinMaxBetween) Write(w io.Writer) {
	w.Write([]byte("func (e E) "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateEMinMaxBetween(f io.Writer, kinds Kinds) {
	minmaxOps := []cmpOp{cmpBinOps[0], cmpBinOps[2]} // Gt and Lt
	minmaxOps[0].name = "Max"
	minmaxOps[1].name = "Min"
	var methods []*InternalEngMinMaxBetween
	for _, bo := range minmaxOps {
		var ks []reflect.Kind
		for _, k := range kinds.Kinds {
			if tc := bo.TypeClass(); tc != nil && tc(k) {
				ks = append(ks, k)
			}
		}
		meth := &InternalEngMinMaxBetween{
			BinOp: bo,
			Kinds: ks,
		}
		methods = append(methods, meth)
	}

	for _, meth := range methods {
		meth.Write(f)
		meth.Iter = true
	}
	for _, meth := range methods {
		meth.Write(f)
	}

}

/* REDUCTION */

type InternalEngReduce struct {
	Kinds []reflect.Kind

	Dim  int // 0 == first dim, -1 == last dim
	Flat bool
}

func (fn *InternalEngReduce) Name() string {
	switch {
	case fn.Flat:
		return "Reduce"
	case fn.Dim == 0:
		return "ReduceFirst"
	case fn.Dim < 0:
		return "ReduceLast"
	case fn.Dim > 0:
		return "ReduceDefault"
	}
	panic("unreachable")
}

func (fn *InternalEngReduce) Signature() *Signature {
	var paramNames, retVals []string
	var paramTemplates, retValTemplates []*template.Template

	switch {
	case fn.Flat:
		paramNames = []string{"t", "a", "defaultValue", "fn"}
		paramTemplates = []*template.Template{reflectType, arrayType, interfaceType, interfaceType}
		retVals = []string{"retVal"}
		retValTemplates = []*template.Template{interfaceType}
	case fn.Dim == 0:
		paramNames = []string{"t", "data", "retVal", "split", "size", "fn"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, intType, intType, interfaceType}
	case fn.Dim < 0:
		paramNames = []string{"t", "data", "retVal", "dimSize", "defaultValue", "fn"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, intType, interfaceType, interfaceType}
	case fn.Dim > 0:
		paramNames = []string{"t", "data", "retVal", "dim0", "dimSize", "outerStride", "stride", "expected", "fn"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, intType, intType, intType, intType, intType, interfaceType}
	}
	return &Signature{
		Name:            fn.Name(),
		NameTemplate:    plainName,
		ParamNames:      paramNames,
		ParamTemplates:  paramTemplates,
		RetVals:         retVals,
		RetValTemplates: retValTemplates,
		Err:             true,
	}
}

func (fn *InternalEngReduce) WriteBody(w io.Writer) {
	var T *template.Template
	switch {
	case fn.Flat:
		T = eReduce
	case fn.Dim == 0:
		T = eReduceFirst
	case fn.Dim < 0:
		T = eReduceLast
	case fn.Dim > 0:
		T = eReduceDefault
	}

	T.Execute(w, fn)
}

func (fn *InternalEngReduce) Write(w io.Writer) {
	w.Write([]byte("func (e E) "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateEReduce(f io.Writer, kinds Kinds) {
	ks := filter(kinds.Kinds, isNotParameterized)

	fn := &InternalEngReduce{
		Kinds: ks,
	}
	fn.Write(f)
	fn.Dim = -1
	fn.Write(f)
	fn.Dim = 1
	fn.Write(f)
	fn.Flat = true
	fn.Write(f)
}

/* UNARY */

type InternalEngUnary struct {
	UnaryOp
	Kinds []reflect.Kind
	Iter  bool
}

func (fn *InternalEngUnary) Signature() *Signature {
	paramNames := []string{"t", "a"}
	paramTemplates := []*template.Template{reflectType, arrayType}

	if fn.Iter {
		paramNames = append(paramNames, "ait")
		paramTemplates = append(paramTemplates, iteratorType)
	}

	if strings.HasPrefix(fn.Name(), "Clamp") {
		paramNames = append(paramNames, "minVal", "maxVal")
		paramTemplates = append(paramTemplates, interfaceType, interfaceType)
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Err:            true,
	}
}

func (fn *InternalEngUnary) Name() string {
	n := fn.UnaryOp.Name()
	if fn.Iter {
		n += "Iter"
	}
	return n
}

func (fn *InternalEngUnary) WriteBody(w io.Writer) {
	var T *template.Template
	switch {
	case fn.Name() == "Clamp":
		T = eUnaryClamp
	case fn.Name() == "ClampIter":
		T = eUnaryClampIter
	case fn.Iter:
		T = eUnaryIter
	default:
		T = eUnary
	}

	T.Execute(w, fn)
}

func (fn *InternalEngUnary) Write(w io.Writer) {
	w.Write([]byte("func (e E) "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateUncondEUnary(f io.Writer, kinds Kinds) {
	var unaries []*InternalEngUnary
	for _, u := range unconditionalUnaries {
		var ks []reflect.Kind
		for _, k := range kinds.Kinds {
			if tc := u.TypeClass(); tc != nil && !tc(k) {
				continue
			}
			ks = append(ks, k)
		}
		ieu := &InternalEngUnary{
			UnaryOp: u,
			Kinds:   ks,
		}
		unaries = append(unaries, ieu)
	}

	for _, u := range unaries {
		u.Write(f)
		u.Iter = true
	}

	for _, u := range unaries {
		u.Write(f)
	}
}

func generateCondEUnary(f io.Writer, kinds Kinds) {
	var unaries []*InternalEngUnary
	for _, u := range conditionalUnaries {
		var ks []reflect.Kind
		for _, k := range kinds.Kinds {
			if tc := u.TypeClass(); tc != nil && !tc(k) {
				continue
			}
			// special case for complex
			if isComplex(k) {
				continue
			}
			ks = append(ks, k)
		}
		ieu := &InternalEngUnary{
			UnaryOp: u,
			Kinds:   ks,
		}
		unaries = append(unaries, ieu)
	}

	for _, u := range unaries {
		u.Write(f)
		u.Iter = true
	}

	for _, u := range unaries {
		u.Write(f)
	}
}

func generateSpecialEUnaries(f io.Writer, kinds Kinds) {
	var unaries []*InternalEngUnary
	for _, u := range specialUnaries {
		var ks []reflect.Kind
		for _, k := range kinds.Kinds {
			if tc := u.TypeClass(); tc != nil && !tc(k) {
				continue
			}

			ks = append(ks, k)
		}

		ieu := &InternalEngUnary{
			UnaryOp: u,
			Kinds:   ks,
		}
		unaries = append(unaries, ieu)
	}

	for _, u := range unaries {
		u.Write(f)
		u.Iter = true
	}

	for _, u := range unaries {
		u.Write(f)
	}
}

/* Argmethods */

type InternalEngArgMethod struct {
	Name   string
	Masked bool
	Flat   bool
	Kinds  []reflect.Kind
}

func (fn *InternalEngArgMethod) Signature() *Signature {
	var name string
	var paramNames []string
	var paramTemplates []*template.Template
	var retVals []string
	var retValTemplates []*template.Template
	var err bool
	switch {
	case fn.Masked && fn.Flat:
		name = fmt.Sprintf("Arg%sFlatMasked", fn.Name)
		paramNames = []string{"t", "a", "mask"}
		paramTemplates = []*template.Template{reflectType, arrayType, boolsType}
		retVals = []string{"retVal"}
		retValTemplates = []*template.Template{intType}
		err = false
	case fn.Masked && !fn.Flat:
		name = fmt.Sprintf("Arg%sIterMasked", fn.Name)
		paramNames = []string{"t", "a", "mask", "it", "lastSize"}
		paramTemplates = []*template.Template{reflectType, arrayType, boolsType, iteratorType, intType}
		retVals = []string{"indices"}
		retValTemplates = []*template.Template{intsType}
		err = true
	case !fn.Masked && fn.Flat:
		name = fmt.Sprintf("Arg%sFlat", fn.Name)
		paramNames = []string{"t", "a"}
		paramTemplates = []*template.Template{reflectType, arrayType}
		retVals = []string{"retVal"}
		retValTemplates = []*template.Template{intType}
		err = false
	default:
		name = fmt.Sprintf("Arg%sIter", fn.Name)
		paramNames = []string{"t", "a", "it", "lastSize"}
		paramTemplates = []*template.Template{reflectType, arrayType, iteratorType, intType}
		retVals = []string{"indices"}
		retValTemplates = []*template.Template{intsType}
		err = true
	}

	return &Signature{
		Name:            name,
		NameTemplate:    plainName,
		ParamNames:      paramNames,
		ParamTemplates:  paramTemplates,
		RetVals:         retVals,
		RetValTemplates: retValTemplates,
		Err:             err,
	}
}

func (fn *InternalEngArgMethod) WriteBody(w io.Writer) {
	switch {
	case fn.Masked && fn.Flat:
		eArgmaxFlatMasked.Execute(w, fn)
	case fn.Masked && !fn.Flat:
		eArgmaxMasked.Execute(w, fn)
	case !fn.Masked && fn.Flat:
		eArgmaxFlat.Execute(w, fn)
	default:
		eArgmax.Execute(w, fn)
	}
}

func (fn *InternalEngArgMethod) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func (e E) "))
	sig.Write(w)
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n"))
}

func generateInternalEngArgmethods(f io.Writer, ak Kinds) {
	ks := filter(ak.Kinds, isOrd)
	meths := []*InternalEngArgMethod{
		&InternalEngArgMethod{Name: "max", Kinds: ks},
		&InternalEngArgMethod{Name: "min", Kinds: ks},
	}

	// default
	for _, fn := range meths {
		fn.Write(f)
		fn.Masked = true
	}
	// masked
	for _, fn := range meths {
		fn.Write(f)
		fn.Flat = true
	}
	// flat masked
	for _, fn := range meths {
		fn.Write(f)
		fn.Flat = true
		fn.Masked = false
	}
	// flat
	for _, fn := range meths {
		fn.Write(f)
	}
}
