package main

import "text/template"

const transTestsRaw = `func gen{{.Name}}Trans[DT internal.OrderedNum](t *testing.T, _ *assert.Assertions) any {
	return func(a, b, c *Dense[DT], sameShape bool) bool {

		_, ok1 := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		weAB := !a.IsNativelyAccessible() || !b.IsNativelyAccessible() || !ok1 || !a.Shape().Eq(b.Shape())
		ab, err := a.{{.Name}}(b)
		if err2, retEarly := qcErrCheck(t, "{{.Name}} - a∙b", a, b, weAB, err); retEarly {
			return err2 == nil
		}

		_, ok2 := b.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		weBC := !b.IsNativelyAccessible() || !c.IsNativelyAccessible() || !ok2 || !b.Shape().Eq(c.Shape())
		bc, err := b.{{.Name}}(c)
		if err, retEarly := qcErrCheck(t, "{{.Name}} - b∙c", b, c, weBC, err); retEarly {
			return err == nil
		}

		ac, err := a.{{.Name}}(c)
		weAC := !a.IsNativelyAccessible() || !b.IsNativelyAccessible() || !ok1 || !a.Shape().Eq(c.Shape())
		if err, retEarly := qcErrCheck(t, "{{.Name}} - a∙c", a, c, weAC, err); retEarly {
			return err == nil
		}

		abD := ab.(tensor.Basic[bool]).Data()
		bcD := bc.(tensor.Basic[bool]).Data()
		acD := ac.(tensor.Basic[bool]).Data()
		for i := range abD {
			if abD[i] && bcD[i] && !acD[i] {
				return false
			}
		}
		return true
	}
}

func gen{{.Name}}TransCisDT[DT internal.OrderedNum](t *testing.T, _ *assert.Assertions) any {
	return func(a, b, c *Dense[DT], sameShape bool) bool {

		_, ok1 := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		weAB := !a.IsNativelyAccessible() || !b.IsNativelyAccessible() || !ok1 || !a.Shape().Eq(b.Shape())
		ab, err := a.{{.Name}}(b, As(a.Dtype()))
		if err2, retEarly := qcErrCheck(t, "{{.Name}} - a∙b", a, b, weAB, err); retEarly {
			return err2 == nil
		}

		_, ok2 := b.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		weBC := !b.IsNativelyAccessible() || !c.IsNativelyAccessible() || !ok2 || !b.Shape().Eq(c.Shape())
		bc, err := b.{{.Name}}(c, As(b.Dtype()))
		if err, retEarly := qcErrCheck(t, "{{.Name}} - b∙c", b, c, weBC, err); retEarly {
			return err == nil
		}

		ac, err := a.{{.Name}}(c, As(a.Dtype()))
		weAC := !a.IsNativelyAccessible() || !b.IsNativelyAccessible() || !ok1 || !a.Shape().Eq(c.Shape())
		if err, retEarly := qcErrCheck(t, "{{.Name}} - a∙c", a, c, weAC, err); retEarly {
			return err == nil
		}

		return !boolNums3Eq(ab.(*Dense[DT]).Data(), bc.(*Dense[DT]).Data(), ac.(*Dense[DT]).Data())
	}
}

`
const denseCmpMethodTestRaw = `func TestDense_{{.Name}}(t *testing.T){
	{{$N := .Name}}
	{{- range $id, $dt := .Datatypes -}}
	qcHelper[{{$dt}}](t, nil, gen{{$N}}Trans[{{$dt}}])
	qcHelper[{{$dt}}](t, nil, gen{{$N}}TransCisDT[{{$dt}}])
	{{end -}}
}
`

var (
	transTests         *template.Template
	denseCmpMethodTest *template.Template
)

func init() {
	transTests = template.Must(template.New("transitivitytests").Funcs(funcs).Parse(transTestsRaw))
	denseCmpMethodTest = template.Must(template.New("denseCmpMethodTests").Funcs(funcs).Parse(denseCmpMethodTestRaw))
}
