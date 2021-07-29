package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

const checkNativeSelectable = `func checkNativeSelectable(t *Dense, axis int, dt dtype.Dtype) error {
	if !t.IsNativelyAccessible() {
		return errors.New("Cannot select on non-natively accessible data")
	}
	if axis >= t.Shape().Dims() && !(t.IsScalar() && axis == 0) {
		return errors.Errorf("Cannot select on axis %d. Shape is %v", axis, t.Shape())
	}
	if t.F() || t.RequiresIterator() {
		return errors.Errorf("Not yet implemented: native select for colmajor or unpacked matrices")
	}
	if t.Dtype() != dt {
		return errors.Errorf("Native selection only works on %v. Got %v", dt, t.Dtype())
	}
	return nil
}
`
const nativeSelectRaw = `
{{- $selName := ( printf "nativeSelect%s" (short .K) ) -}}
{{- if .N -}}
	{{- $selName = ( printf "Select%s" (short .K) ) -}}
{{- end -}}

// {{$selName}} creates a slice of flat data types. See Example of NativeSelectF64.
func {{$selName}}(t *Dense, axis int) (retVal [][]{{asType .K}}, err error) {
	if err := checkNativeSelectable(t, axis, {{reflectKind .K}}); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]{{asType .K}}, 1)
		retVal[0] = t.{{sliceOf .K}}
	case 2:
		if axis == 0 {
			return {{if .N}}Matrix{{short .K}}{{else}}nativeDenseMatrix{{short .K}}{{end}}(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.{{sliceOf .K}}
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]{{asType .K}}, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]{{asType .K}}, 0)
			hdr := (*reflect.SliceHeader)(unsafe.Pointer(&s))
			hdr.Data = uintptr(unsafe.Pointer(&data[i]))
			hdr.Len = stride
			hdr.Cap = stride
			retVal = append(retVal, s)
			r++
		}
		return retVal, nil

	}
	return
}
`
const nativeSelectTestRaw = `
{{- $selName := ( printf "nativeSelect%s" (short .K) ) -}}
{{- if .N -}}
	{{- $selName = ( printf "Select%s" (short .K) ) -}}
{{- end -}}
func Test{{$selName}}(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	var err error
	var x [][]{{asType .K}}
	T = New(Of({{reflectKind .K}}), WithShape(2, 3, 4, 5), )
	if x, err = {{$selName}}(T, 1); err != nil {
		t.Fatal(err)
	}
	assert.Equal(6, len(x))
	assert.Equal(20, len(x[0]))

	T = New(Of({{reflectKind .K}}), WithShape(2, 3, 4, 5), )
	if x, err = {{$selName}}(T, 0); err != nil {
		t.Fatal(err)
	}
	assert.Equal(2, len(x))
	assert.Equal(60, len(x[0]))

	T = New(Of({{reflectKind .K}}), WithShape(2, 3, 4, 5), )
	if x, err = {{$selName}}(T, 3); err != nil {
		t.Fatal(err)
	}
	assert.Equal(120, len(x))
	assert.Equal(1, len(x[0]))

	T = New(Of({{reflectKind .K}}), WithShape(2, 3), )
	if x, err = {{$selName}}(T, 0); err != nil {
		t.Fatal(err)
	}
	assert.Equal(2, len(x))
	assert.Equal(3, len(x[0]))

	T = New(Of({{reflectKind .K}}), WithShape(2, 3), )
	if x, err = {{$selName}}(T, 1); err != nil {
		t.Fatal(err)
	}
	assert.Equal(6, len(x))
	assert.Equal(1, len(x[0]))

	T = New(FromScalar({{if eq .K.String "bool" -}}false{{else if eq .K.String "string" -}}""{{else -}}{{asType .K}}(0) {{end -}} ))
	if x, err = {{$selName}}(T, 0); err != nil {
		t.Fatal(err)
	}
	assert.Equal(1, len(x))
	assert.Equal(1, len(x[0]))

	if _, err = {{$selName}}(T, 10); err == nil{
		t.Fatal("Expected errors")
	}
}
`

const nativeSelectStubsRaw = `//go:linkname Select{{short .}} gorgonia.org/tensor.nativeSelect{{short .}}

// Select{{short .}} creates a slice of {{asType .}}s. See Example of NativeSelectF64.
func Select{{short .}}(t *tensor.Dense, axis int) (retVal [][]{{asType .}}, err error)
`

var (
	NativeSelect      *template.Template
	NativeSelectTest  *template.Template
	NativeSelectStubs *template.Template
)

func init() {
	NativeSelect = template.Must(template.New("NativeSelect").Funcs(funcs).Parse(nativeSelectRaw))
	NativeSelectTest = template.Must(template.New("NativeSelectTest").Funcs(funcs).Parse(nativeSelectTestRaw))
	NativeSelectStubs = template.Must(template.New("NativeSelectStub").Funcs(funcs).Parse(nativeSelectStubsRaw))
}

// generateNativeSelect generates code for the native selection. `isNative` indicates if the
// code is meant to be generated for the native package. The code is generated for the native package
// only for the purposes of the `purego` build tag.
func generateNativeSelect(isNative bool) func(io.Writer, Kinds) {
	type IterTup struct {
		N bool
		K reflect.Kind
	}
	return func(f io.Writer, ak Kinds) {
		if isNative {
			fmt.Fprintf(f, importUnqualifiedTensor)
		} else {
			fmt.Fprintf(f, "%v\n", checkNativeSelectable)
		}
		ks := filter(ak.Kinds, isSpecialized)
		for _, k := range ks {
			fmt.Fprintf(f, "/* Native Select for %v */\n\n", k)
			NativeSelect.Execute(f, IterTup{N: isNative, K: k})
			fmt.Fprint(f, "\n\n")
		}
	}
}

func generateNativeSelectTests(isNative bool) func(f io.Writer, ak Kinds) {
	type IterTup struct {
		N bool
		K reflect.Kind
	}
	return func(f io.Writer, ak Kinds) {
		if isNative {
			fmt.Fprintf(f, importUnqualifiedTensor)
		}
		ks := filter(ak.Kinds, isSpecialized)
		for _, k := range ks {
			NativeSelectTest.Execute(f, IterTup{N: isNative, K: k})
			fmt.Fprint(f, "\n\n")
		}
	}
}

func generateNativeSelectStubs(f io.Writer, ak Kinds) {
	fmt.Fprintf(f, importUnsafe) // this is required for go:linkname to work
	ks := filter(ak.Kinds, isSpecialized)
	for _, k := range ks {
		NativeSelectStubs.Execute(f, k)
		fmt.Fprintf(f, "\n\n")
	}
}

func generateNativeSelChecks(f io.Writer, ak Kinds) {
	// fmt.Fprintf(f, importUnqualifiedTensor)  // already generated by generateNativeIterChecks
	fmt.Fprintf(f, "%v\n", checkNativeSelectable)
}
