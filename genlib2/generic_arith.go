package main

import (
	"fmt"
	"io"
	"strings"
	"text/template"
)

type GenericVecVecArith struct {
	TypedBinOp
	Iter          bool
	Incr          bool
	WithRecv      bool      // not many BinOps have this
	Check         TypeClass // can be nil
	CheckTemplate string
}

func (fn *GenericVecVecArith) Name() string {
	switch {
	case fn.Iter && fn.Incr:
		return fmt.Sprintf("%sIterIncr", fn.TypedBinOp.Name())
	case fn.Iter && !fn.Incr:
		return fmt.Sprintf("%sIter", fn.TypedBinOp.Name())
	case !fn.Iter && fn.Incr:
		return fmt.Sprintf("%sIncr", fn.TypedBinOp.Name())
	case fn.WithRecv:
		return fmt.Sprintf("%vRecv", fn.TypedBinOp.Name())
	default:
		return fmt.Sprintf("Vec%s", fn.TypedBinOp.Name())
	}
}

func (fn *GenericVecVecArith) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	var err bool

	switch {
	case fn.Iter && fn.Incr:
		paramNames = []string{"a", "b", "incr", "ait", "bit", "iit"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType, iteratorType, iteratorType, iteratorType}
		err = true
	case fn.Iter && !fn.Incr:
		paramNames = []string{"a", "b", "ait", "bit"}
		paramTemplates = []*template.Template{sliceType, sliceType, iteratorType, iteratorType}
		err = true
	case !fn.Iter && fn.Incr:
		paramNames = []string{"a", "b", "incr"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType}
	case fn.WithRecv:
		paramNames = []string{"a", "b", "recv"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType}
	default:
		paramNames = []string{"a", "b"}
		paramTemplates = []*template.Template{sliceType, sliceType}
	}

	if fn.Check != nil {
		err = true
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   typeAnnotatedName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Kind: fn.Kind(),
		Err:  err,
	}
}

func (fn *GenericVecVecArith) WriteBody(w io.Writer) {
	var Range, Left, Right string
	var Index0, Index1, Index2 string
	var IterName0, IterName1, IterName2 string
	var T *template.Template

	Range = "a"
	Index0 = "i"
	Index1 = "j"
	Left = "a[i]"
	Right = "b[j]"

	T = template.New(fn.Name()).Funcs(funcs)
	switch {
	case fn.Iter && fn.Incr:
		Range = "incr"
		Index2 = "k"
		IterName0 = "ait"
		IterName1 = "bit"
		IterName2 = "iit"
		T = template.Must(T.Parse(genericTernaryIterLoopRaw))
		template.Must(T.New("loopbody").Parse(iterIncrLoopBody))
	case fn.Iter && !fn.Incr:
		IterName0 = "ait"
		IterName1 = "bit"
		T = template.Must(T.Parse(genericBinaryIterLoopRaw))
		template.Must(T.New("loopbody").Parse(basicSet))
	case !fn.Iter && fn.Incr:
		Range = "incr"
		Right = "b[i]"
		T = template.Must(T.Parse(genericLoopRaw))
		template.Must(T.New("loopbody").Parse(basicIncr))
	case fn.WithRecv:
		Range = "recv"
		Right = "b[i]"
		T = template.Must(T.Parse(genericLoopRaw))
		template.Must(T.New("loopbody").Parse(basicSet))
	default:
		Right = "b[i]"
		T = template.Must(T.Parse(genericLoopRaw))
		template.Must(T.New("loopbody").Parse(basicSet))
	}
	template.Must(T.New("callFunc").Parse(binOpCallFunc))
	template.Must(T.New("opDo").Parse(binOpDo))
	template.Must(T.New("symbol").Parse(fn.SymbolTemplate()))

	if fn.Check != nil && fn.Check(fn.Kind()) {
		w.Write([]byte("var errs errorIndices\n"))
	}
	template.Must(T.New("check").Parse(fn.CheckTemplate))

	lb := LoopBody{
		TypedOp: fn.TypedBinOp,
		Range:   Range,
		Left:    Left,
		Right:   Right,

		Index0: Index0,
		Index1: Index1,
		Index2: Index2,

		IterName0: IterName0,
		IterName1: IterName1,
		IterName2: IterName2,
	}
	T.Execute(w, lb)
}

func (fn *GenericVecVecArith) Write(w io.Writer) {
	sig := fn.Signature()
	if !fn.Iter && isFloat(fn.Kind()) && !fn.WithRecv {
		// golinkPragma.Execute(w, fn)
		w.Write([]byte("func "))
		sig.Write(w)
		if fn.Incr {
			fmt.Fprintf(w, "{ %v%v(a, b, incr)}\n", vecPkg(fn.Kind()), getalias(fn.Name()))
		} else {
			fmt.Fprintf(w, "{ %v%v(a, b)}\n", vecPkg(fn.Kind()), getalias(fn.Name()))
		}
		return
	}

	w.Write([]byte("func "))
	sig.Write(w)

	switch {
	case !fn.Iter && fn.Incr:
		w.Write([]byte("{\na = a[:len(a)]; b = b[:len(a)]; incr = incr[:len(a)]\n"))
	case fn.WithRecv:
		w.Write([]byte("{\na = a[:len(recv)]; b = b[:len(recv)]\n"))
	case !fn.Iter && !fn.Incr && !fn.WithRecv:
		w.Write([]byte("{\na = a[:len(a)]; b = b[:len(a)]\n"))
	default:
		w.Write([]byte("{\n"))
	}
	fn.WriteBody(w)
	if sig.Err {
		if fn.Check != nil {
			w.Write([]byte("\nif err != nil {\n return\n}\nif len(errs) > 0 {\n return errs }\nreturn nil"))
		} else {
			w.Write([]byte("\nreturn\n"))
		}
	}
	w.Write([]byte("}\n\n"))
}

type GenericMixedArith struct {
	GenericVecVecArith
	LeftVec bool
}

func (fn *GenericMixedArith) Name() string {
	n := fn.GenericVecVecArith.Name()
	n = strings.TrimPrefix(n, "Vec")
	if fn.LeftVec {
		n += "VS"
	} else {
		n += "SV"
	}
	return n
}

func (fn *GenericMixedArith) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	var err bool

	switch {
	case fn.Iter && fn.Incr:
		paramNames = []string{"a", "b", "incr", "ait", "iit"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType, iteratorType, iteratorType}
		if fn.LeftVec {
			paramTemplates[1] = scalarType
		} else {
			paramTemplates[0] = scalarType
			paramNames[3] = "bit"
		}
		err = true
	case fn.Iter && !fn.Incr:
		paramNames = []string{"a", "b", "ait"}
		paramTemplates = []*template.Template{sliceType, sliceType, iteratorType}
		if fn.LeftVec {
			paramTemplates[1] = scalarType
		} else {
			paramTemplates[0] = scalarType
			paramNames[2] = "bit"
		}

		err = true
	case !fn.Iter && fn.Incr:
		paramNames = []string{"a", "b", "incr"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType}
		if fn.LeftVec {
			paramTemplates[1] = scalarType
		} else {
			paramTemplates[0] = scalarType
		}

	default:
		paramNames = []string{"a", "b"}
		paramTemplates = []*template.Template{sliceType, sliceType}
		if fn.LeftVec {
			paramTemplates[1] = scalarType
		} else {
			paramTemplates[0] = scalarType
		}
	}

	if fn.Check != nil {
		err = true
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   typeAnnotatedName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Kind: fn.Kind(),
		Err:  err,
	}
}

func (fn *GenericMixedArith) WriteBody(w io.Writer) {
	var Range, Left, Right string
	var Index0, Index1 string
	var IterName0, IterName1 string

	Range = "a"
	Left = "a[i]"
	Right = "b[i]"
	Index0 = "i"

	T := template.New(fn.Name()).Funcs(funcs)
	switch {
	case fn.Iter && fn.Incr:
		Range = "incr"
		T = template.Must(T.Parse(genericBinaryIterLoopRaw))
		template.Must(T.New("loopbody").Parse(iterIncrLoopBody))
	case fn.Iter && !fn.Incr:
		T = template.Must(T.Parse(genericUnaryIterLoopRaw))
		template.Must(T.New("loopbody").Parse(basicSet))
	case !fn.Iter && fn.Incr:
		Range = "incr"
		T = template.Must(T.Parse(genericLoopRaw))
		template.Must(T.New("loopbody").Parse(basicIncr))
	default:
		T = template.Must(T.Parse(genericLoopRaw))
		template.Must(T.New("loopbody").Parse(basicSet))
	}

	if fn.LeftVec {
		Right = "b"
	} else {
		Left = "a"
		if !fn.Incr {
			Range = "b"
		}
		// Index0 = "j"
	}

	switch {
	case fn.Iter && fn.Incr && fn.LeftVec:
		IterName0 = "ait"
		IterName1 = "iit"
		Index1 = "k"
	case fn.Iter && !fn.Incr && fn.LeftVec:
		IterName0 = "ait"
	case fn.Iter && fn.Incr && !fn.LeftVec:
		IterName0 = "bit"
		IterName1 = "iit"
		Index1 = "k"
	case fn.Iter && !fn.Incr && !fn.LeftVec:
		IterName0 = "bit"
	}

	template.Must(T.New("callFunc").Parse(binOpCallFunc))
	template.Must(T.New("opDo").Parse(binOpDo))
	template.Must(T.New("symbol").Parse(fn.SymbolTemplate()))

	if fn.Check != nil && fn.Check(fn.Kind()) {
		w.Write([]byte("var errs errorIndices\n"))
	}
	template.Must(T.New("check").Parse(fn.CheckTemplate))

	lb := LoopBody{
		TypedOp: fn.TypedBinOp,
		Range:   Range,
		Left:    Left,
		Right:   Right,

		Index0:    Index0,
		Index1:    Index1,
		IterName0: IterName0,
		IterName1: IterName1,
	}
	T.Execute(w, lb)
}

func (fn *GenericMixedArith) Write(w io.Writer) {
	sig := fn.Signature()

	w.Write([]byte("func "))
	sig.Write(w)

	w.Write([]byte("{\n"))

	fn.WriteBody(w)
	if sig.Err {
		if fn.Check != nil {
			w.Write([]byte("\nif err != nil {\n return\n}\nif len(errs) > 0 {\n return errs }\nreturn nil"))
		} else {
			w.Write([]byte("\nreturn\n"))
		}
	}
	w.Write([]byte("}\n\n"))
}

type GenericScalarScalarArith struct {
	TypedBinOp
}

func (fn *GenericScalarScalarArith) Signature() *Signature {
	return &Signature{
		Name:            fn.Name(),
		NameTemplate:    typeAnnotatedName,
		ParamNames:      []string{"a", "b"},
		ParamTemplates:  []*template.Template{scalarType, scalarType},
		RetVals:         []string{""},
		RetValTemplates: []*template.Template{scalarType},
		Kind:            fn.Kind(),
	}
}

func (fn *GenericScalarScalarArith) WriteBody(w io.Writer) {
	tmpl := `return {{if .IsFunc -}}
			{{ template "callFunc" . -}}
		{{else -}}
			{{template "opDo" . -}}
		{{end -}}`
	opDo := `a {{template "symbol" .Kind}} b`
	callFunc := `{{template "symbol" .Kind}}(a, b)`

	T := template.Must(template.New(fn.Name()).Funcs(funcs).Parse(tmpl))
	template.Must(T.New("opDo").Parse(opDo))
	template.Must(T.New("callFunc").Parse(callFunc))
	template.Must(T.New("symbol").Parse(fn.SymbolTemplate()))

	T.Execute(w, fn)
}

func (fn *GenericScalarScalarArith) Write(w io.Writer) {
	w.Write([]byte("func "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{"))
	fn.WriteBody(w)
	w.Write([]byte("}\n"))
}

func makeGenericVecVecAriths(tbo []TypedBinOp) (retVal []*GenericVecVecArith) {
	for _, tb := range tbo {
		if tc := tb.TypeClass(); tc != nil && !tc(tb.Kind()) {
			continue
		}
		fn := &GenericVecVecArith{
			TypedBinOp: tb,
		}
		if tb.Name() == "Div" && !isFloatCmplx(tb.Kind()) {
			fn.Check = panicsDiv0
			fn.CheckTemplate = check0
		}

		retVal = append(retVal, fn)

	}

	return retVal
}

func makeGenericMixedAriths(tbo []TypedBinOp) (retVal []*GenericMixedArith) {
	for _, tb := range tbo {
		if tc := tb.TypeClass(); tc != nil && !tc(tb.Kind()) {
			continue
		}
		fn := &GenericMixedArith{
			GenericVecVecArith: GenericVecVecArith{
				TypedBinOp: tb,
			},
		}
		if tb.Name() == "Div" && !isFloatCmplx(tb.Kind()) {
			fn.Check = panicsDiv0
			fn.CheckTemplate = check0
		}
		retVal = append(retVal, fn)
	}
	return
}

func makeGenericScalarScalarAriths(tbo []TypedBinOp) (retVal []*GenericScalarScalarArith) {
	for _, tb := range tbo {
		if tc := tb.TypeClass(); tc != nil && !tc(tb.Kind()) {
			continue
		}
		fn := &GenericScalarScalarArith{
			TypedBinOp: tb,
		}
		retVal = append(retVal, fn)
	}
	return
}

func generateGenericVecVecArith(f io.Writer, ak Kinds) {
	gen := makeGenericVecVecAriths(typedAriths)

	// importStmt := `
	// import (
	// 	_ "unsafe"

	// _ "gorgonia.org/vecf32"
	// _ "gorgonia.org/vecf64")
	// `
	// f.Write([]byte(importStmt))

	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = false
		g.Iter = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)
	}
	for _, g := range gen {
		g.Incr = false
		g.Iter = false
		g.WithRecv = true
		g.Write(f)
	}
}

func generateGenericMixedArith(f io.Writer, ak Kinds) {
	gen := makeGenericMixedAriths(typedAriths)

	// SV first
	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = false
		g.Iter = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)

		// reset
		g.LeftVec = true
		g.Incr = false
		g.Iter = false
	}

	// VS
	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = false
		g.Iter = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)
	}
}

func generateGenericScalarScalarArith(f io.Writer, ak Kinds) {
	gen := makeGenericScalarScalarAriths(typedAriths)
	for _, g := range gen {
		g.Write(f)
	}
}
