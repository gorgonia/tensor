package main

import (
	"fmt"
	"io"
	"text/template"
)

const checkNativeiterable = `func checkNativeIterable(t *Dense, dims int, dt Dtype) error {
	// checks:
	if !t.IsNativelyAccessible() {
		return errors.Errorf("Cannot convert *Dense to *mat.Dense. Data is inaccessible")
	}

	if t.shape.Dims() != dims {
		return errors.Errorf("Cannot convert *Dense to native iterator. Expected number of dimension: %d, T has got %d dimensions (Shape: %v)", dims, t.Dims(), t.Shape())
	}

	if t.DataOrder().isColMajor() || t.RequiresIterator() {
		return errors.Errorf("Not yet implemented: native matrix for colmajor or unpacked matrices")
	}

	if t.Dtype() != dt {
		return errors.Errorf("Conversion to native iterable only works on %v. Got %v", dt, t.Dtype())
	}

	return nil
}
`

const nativeIterRaw = `// NativeVector{{short .}} converts a *Dense into a []{{asType .}}
// If the *Dense does not represent a vector of the wanted type, it will return an error.
func NativeVector{{short .}}(t *Dense) (retVal []{{asType .}}, err error) {
	if err = checkNativeIterable(t, 1, {{reflectKind .}}); err != nil {
		return nil, err
	}
	return t.{{sliceOf .}}, nil
}

// NativeMatrix{{short .}} converts a  *Dense into a [][]{{asType .}}
// If the *Dense does not represent a matrix of the wanted type, it will return an error.
func NativeMatrix{{short .}}(t *Dense) (retVal [][]{{asType .}}, err error) {
	if err = checkNativeIterable(t, 2, {{reflectKind .}}); err != nil {
		return nil, err
	}

	data := t.{{sliceOf .}}
	shape := t.Shape()
	strides := t.Strides()

	rows := shape[0]
	cols := shape[1]
	rowStride := strides[0]
	retVal = make([][]{{asType .}}, rows)
	for i := range retVal {
		start := i * rowStride
		hdr := &reflect.SliceHeader{
			Data: uintptr(unsafe.Pointer(&data[start])),
			Len:  cols,
			Cap:  cols,
		}
		retVal[i] = *(*[]{{asType .}})(unsafe.Pointer(hdr))
	}
	return
}

// Native3Tensor{{short .}} converts a *Dense into  a [][][]{{asType .}}. 
// If the *Dense does not represent a 3-tensor of the wanted type, it will return an error.
func Native3Tensor{{short .}}(t *Dense) (retVal [][][]{{asType .}}, err error) {
	if err = checkNativeIterable(t, 3, {{reflectKind .}}); err != nil {
		return nil, err
	}

	data := t.{{sliceOf .}}
	shape := t.Shape()
	strides := t.Strides()

	layers := shape[0]
	rows := shape[1]
	cols := shape[2]
	layerStride := strides[0]
	rowStride := strides[1]
	retVal = make([][][]{{asType .}}, layers)
	for i := range retVal {
		retVal[i] = make([][]{{asType .}}, rows)
		for j := range retVal[i] {
			start := i*layerStride + j*rowStride
			hdr := &reflect.SliceHeader{
				Data: uintptr(unsafe.Pointer(&data[start])),
				Len:  cols,
				Cap:  cols,
			}
			retVal[i][j] = *(*[]{{asType .}})(unsafe.Pointer(hdr))
		}
	}
	return
}
`

const nativeIterTestRaw = `func Test_NativeVector{{short .}}(t *testing.T) {
	assert := assert.New(t)
	{{if isRangeable . -}}
	T := New(WithBacking(Range({{reflectKind .}}, 0, 6)), WithShape(6))
	{{else -}}
	T := New(Of({{reflectKind .}}), WithShape(6))
	{{end -}}
	it, err := NativeVector{{short .}}(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_NativeMatrix{{short .}}(t *testing.T) {
	assert := assert.New(t)
	{{if isRangeable . -}}
	T := New(WithBacking(Range({{reflectKind .}}, 0, 6)), WithShape(2, 3))
	{{else -}}
	T := New(Of({{reflectKind .}}), WithShape(2, 3))
	{{end -}}
	it, err := NativeMatrix{{short .}}(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func TestNative3Tensor{{short .}}(t *testing.T) {
	assert := assert.New(t)
	{{if isRangeable . -}}
	T := New(WithBacking(Range({{reflectKind .}}, 0, 24)), WithShape(2, 3, 4))
	{{else -}}
	T := New(Of({{reflectKind .}}), WithShape(2, 3, 4))
	{{end -}}
	it, err := Native3Tensor{{short .}}(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}
`

var (
	NativeIter     *template.Template
	NativeIterTest *template.Template
)

func init() {
	NativeIter = template.Must(template.New("NativeIter").Funcs(funcs).Parse(nativeIterRaw))
	NativeIterTest = template.Must(template.New("NativeIterTest").Funcs(funcs).Parse(nativeIterTestRaw))
}

func generateNativeIterators(f io.Writer, ak Kinds) {
	fmt.Fprintf(f, "%v\n", checkNativeiterable)
	ks := filter(ak.Kinds, isSpecialized)
	for _, k := range ks {
		fmt.Fprintf(f, "/* Native Iterables for %v */\n\n", k)
		NativeIter.Execute(f, k)
		fmt.Fprint(f, "\n\n")
	}
}

func generateNativeIteratorTests(f io.Writer, ak Kinds) {
	ks := filter(ak.Kinds, isSpecialized)
	for _, k := range ks {
		NativeIterTest.Execute(f, k)
		fmt.Fprint(f, "\n\n")
	}
}
