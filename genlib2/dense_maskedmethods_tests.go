package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

type MaskCmpMethodTest struct {
	Kind reflect.Kind
	Name string
}

const testMaskCmpMethodRaw = `func TestDense_{{title .Name}}_{{short .Kind}}(t *testing.T){
    assert := assert.New(t)
    T := New(Of({{reflectKind .Kind}}), WithShape(2, 3, 4, 5))
    assert.False(T.IsMasked())
    data := T.{{sliceOf .Kind}}
    for i := range data {
{{if eq "string" (asType .Kind) -}}
		data[i] = fmt.Sprint(i)
{{else -}}
		data[i] = {{asType .Kind}}(i)
{{end -}}
	}
{{if eq "string" (asType .Kind) -}}
    T.MaskedEqual(fmt.Sprint(0))
{{else -}}
    T.MaskedEqual({{asType .Kind}}(0))
{{end -}}
	assert.True(T.IsMasked())
{{if eq "string" (asType .Kind) -}}
	T.MaskedEqual(fmt.Sprint(1))
{{else -}}
	T.MaskedEqual({{asType .Kind}}(1))
{{end -}}
	assert.True(T.mask[0] && T.mask[1])
{{if eq "string" (asType .Kind) -}}
	T.MaskedNotEqual(fmt.Sprint(2))
{{else -}}
	T.MaskedNotEqual({{asType .Kind}}(2))
{{end -}}
	assert.False(T.mask[2] && !(T.mask[0]))

    T.ResetMask()
{{if eq "string" (asType .Kind) -}}
	T.MaskedInside(fmt.Sprint(1), fmt.Sprint(22))
{{else -}}
	T.MaskedInside({{asType .Kind}}(1), {{asType .Kind}}(22))
{{end -}}
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
{{if eq "string" (asType .Kind) -}}
	T.MaskedOutside(fmt.Sprint(1), fmt.Sprint(22))
{{else -}}
	T.MaskedOutside({{asType .Kind}}(1), {{asType .Kind}}(22))
{{end -}}
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

    T.ResetMask()
    for i := 0; i < 5; i++ {
{{if eq "string" (asType .Kind) -}}
		T.MaskedEqual(fmt.Sprint(i*10))
{{else -}}
		T.MaskedEqual({{asType .Kind}}(i*10))
{{end -}}
	}
    it := IteratorFromDense(T)

    j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5,j)
    }
    `

var (
	testMaskCmpMethod *template.Template
)

func init() {
	testMaskCmpMethod = template.Must(template.New("testmaskcmpmethod").Funcs(funcs).Parse(testMaskCmpMethodRaw))
}

func generateMaskCmpMethodsTests(f io.Writer, generic Kinds) {
	for _, mm := range maskcmpMethods {
		fmt.Fprintf(f, "/* %s */ \n\n", mm.Name)
		for _, k := range generic.Kinds {
			if isOrd(k) {
				if mm.ReqFloat && isntFloat(k) {

				} else {
					op := MaskCmpMethodTest{k, mm.Name}
					testMaskCmpMethod.Execute(f, op)
				}
			}
		}
	}
}
