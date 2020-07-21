package main

import (
	"fmt"
	"io"
	"text/template"
)

const checkNativeSelectable = `func checkNativeSelectable(t *Dense, axis int, dt Dtype) error {
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
const nativeSelectRaw = `// Select{{short .}} creates a slice of flat data types. See Example of NativeSelectF64.
func Select{{short .}}(t *Dense, axis int) (retVal [][]{{asType .}}, err error) {
	if err := checkNativeSelectable(t, axis, {{reflectKind .}}); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]{{asType .}}, 1)
		retVal[0] = t.{{sliceOf .}}
	case 2:
		if axis == 0 {
			return Matrix{{short .}}(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.{{sliceOf .}}
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]{{asType .}}, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]{{asType .}}, 0)
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
const nativeSelectTestRaw = `func TestSelect{{short .}}(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	var err error
	var x [][]{{asType .}}
	T = New(Of({{reflectKind .}}), WithShape(2, 3, 4, 5), )
	if x, err = Select{{short .}}(T, 1); err != nil {
		t.Fatal(err)
	}
	assert.Equal(6, len(x))
	assert.Equal(20, len(x[0]))

	T = New(Of({{reflectKind .}}), WithShape(2, 3, 4, 5), )
	if x, err = Select{{short .}}(T, 0); err != nil {
		t.Fatal(err)
	}
	assert.Equal(2, len(x))
	assert.Equal(60, len(x[0]))

	T = New(Of({{reflectKind .}}), WithShape(2, 3, 4, 5), )
	if x, err = Select{{short .}}(T, 3); err != nil {
		t.Fatal(err)
	}
	assert.Equal(120, len(x))
	assert.Equal(1, len(x[0]))

	T = New(Of({{reflectKind .}}), WithShape(2, 3), )
	if x, err = Select{{short .}}(T, 0); err != nil {
		t.Fatal(err)
	}
	assert.Equal(2, len(x))
	assert.Equal(3, len(x[0]))

	T = New(Of({{reflectKind .}}), WithShape(2, 3), )
	if x, err = Select{{short .}}(T, 1); err != nil {
		t.Fatal(err)
	}
	assert.Equal(6, len(x))
	assert.Equal(1, len(x[0]))

	T = New(FromScalar({{if eq .String "bool" -}}false{{else if eq .String "string" -}}""{{else -}}{{asType .}}(0) {{end -}} ))
	if x, err = Select{{short .}}(T, 0); err != nil {
		t.Fatal(err)
	}
	assert.Equal(1, len(x))
	assert.Equal(1, len(x[0]))

	if _, err = Select{{short .}}(T, 10); err == nil{
		t.Fatal("Expected errors")
	}
}
`

var (
	NativeSelect     *template.Template
	NativeSelectTest *template.Template
)

func init() {
	NativeSelect = template.Must(template.New("NativeSelect").Funcs(funcs).Parse(nativeSelectRaw))
	NativeSelectTest = template.Must(template.New("NativeSelectTest").Funcs(funcs).Parse(nativeSelectTestRaw))
}

func generateNativeSelect(f io.Writer, ak Kinds) {
	fmt.Fprintf(f, importUnqualifiedTensor)
	fmt.Fprintf(f, "%v\n", checkNativeSelectable)
	ks := filter(ak.Kinds, isSpecialized)
	for _, k := range ks {
		fmt.Fprintf(f, "/* Native Select for %v */\n\n", k)
		NativeSelect.Execute(f, k)
		fmt.Fprint(f, "\n\n")
	}
}

func generateNativeSelectTests(f io.Writer, ak Kinds) {
	fmt.Fprintf(f, importUnqualifiedTensor)
	ks := filter(ak.Kinds, isSpecialized)
	for _, k := range ks {
		NativeSelectTest.Execute(f, k)
		fmt.Fprint(f, "\n\n")
	}
}
