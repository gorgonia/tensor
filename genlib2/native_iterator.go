package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

const checkNativeiterable = `func checkNativeIterable(t *Dense, dims int, dt dtype.Dtype) error {
	// checks:
	if !t.IsNativelyAccessible() {
		return errors.Errorf("Cannot convert *Dense to *mat.Dense. Data is inaccessible")
	}

	if t.Shape().Dims() != dims {
		return errors.Errorf("Cannot convert *Dense to native iterator. Expected number of dimension: %d, T has got %d dimensions (Shape: %v)", dims, t.Dims(), t.Shape())
	}

	if t.F() || t.RequiresIterator() {
		return errors.Errorf("Not yet implemented: native matrix for colmajor or unpacked matrices")
	}

	if t.Dtype() != dt {
		return errors.Errorf("Conversion to native iterable only works on %v. Got %v", dt, t.Dtype())
	}

	return nil
}
`

const nativeIterRaw = `
{{- $vecName := ( printf "nativeDenseVector%s" (short .K) ) -}}
{{- $matName := ( printf "nativeDenseMatrix%s" (short .K) ) -}}
{{- $T3Name := ( printf "nativeDenseTensor3%s" (short .K) ) -}}
{{- if .N -}}
	{{- $vecName = ( printf "Vector%s" (short .K) ) -}}
	{{- $matName = ( printf "Matrix%s" (short .K) ) -}}
	{{- $T3Name = ( printf "Tensor3%s" (short .K) ) -}}
{{- end -}}

// {{$vecName}} converts a *Dense into a []{{asType .K}}
// If the *Dense does not represent a vector of the wanted type, it will return
// an error.
func {{$vecName}}(t *Dense) (retVal []{{asType .K}}, err error) {
	if err = checkNativeIterable(t, 1, {{reflectKind .K}}); err != nil {
		return nil, err
	}
	return t.{{sliceOf .K}}, nil
}

// {{$matName}} converts a  *Dense into a [][]{{asType .K}}
// If the *Dense does not represent a matrix of the wanted type, it
// will return an error.
func {{$matName}}(t *Dense) (retVal [][]{{asType .K}}, err error) {
	if err = checkNativeIterable(t, 2, {{reflectKind .K}}); err != nil {
		return nil, err
	}

	data := t.{{sliceOf .K}}
	shape := t.Shape()
	strides := t.Strides()

	rows := shape[0]
	cols := shape[1]
	rowStride := strides[0]
	retVal = make([][]{{asType .K}}, rows)
	for i := range retVal {
		start := i * rowStride
		retVal[i] = make([]{{asType .K}}, 0)
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&retVal[i]))
		hdr.Data = uintptr(unsafe.Pointer(&data[start]))
		hdr.Cap = cols
		hdr.Len = cols
	}
	return
}

// {{$T3Name}} converts a *Dense into a  [][][]{{asType .K}}.
// If the *Dense does not represent a 3-tensor of the wanted type, it will return an error.
func {{$T3Name}}(t *Dense) (retVal [][][]{{asType .K}}, err error) {
	if err = checkNativeIterable(t, 3, {{reflectKind .K}}); err != nil {
		return nil, err
	}

	data := t.{{sliceOf .K}}
	shape := t.Shape()
	strides := t.Strides()

	layers := shape[0]
	rows := shape[1]
	cols := shape[2]
	layerStride := strides[0]
	rowStride := strides[1]
	retVal = make([][][]{{asType .K}}, layers)
	for i := range retVal {
		retVal[i] = make([][]{{asType .K}}, rows)
		for j := range retVal[i] {
			retVal[i][j] = make([]{{asType .K}}, 0)
			start := i*layerStride + j*rowStride
			hdr := (*reflect.SliceHeader)(unsafe.Pointer(&retVal[i][j]))
			hdr.Data = uintptr(unsafe.Pointer(&data[start]))
			hdr.Cap = cols
			hdr.Len = cols
		}
	}
	return
}
`

const nativeIterStubsRaw = `//go:linkname  Vector{{short .}} gorgonia.org/tensor.nativeDenseVector{{short .}}

// Vector{{short .}} converts a *Dense into a []{{asType .}}
// If the *Dense does not represent a vector of the wanted type, it will return
// an error.
func Vector{{short .}}(t *tensor.Dense) (retVal []{{asType .}}, err error)

//go:linkname Matrix{{short .}} gorgonia.org/tensor.nativeDenseMatrix{{short .}}

// Matrix{{short .}} converts a  *Dense into a [][]{{asType .}}
// If the *Dense does not represent a matrix of the wanted type, it
// will return an error.
func Matrix{{short .}}(t *tensor.Dense) (retVal [][]{{asType .}}, err error)

//go:linkname Tensor3{{short .}} gorgonia.org/tensor.nativeDenseTensor3{{short .}}

// Tensor3{{short .}} converts a *Dense into a  [][][]{{asType .}}.
// If the *Dense does not represent a 3-tensor of the wanted type, it will return an error.
func Tensor3{{short .}}(t *tensor.Dense) (retVal [][][]{{asType .}}, err error)
`

const nativeIterTestRaw = `
{{- $pkgTVecName := ( printf "nativeDenseVector%s" (short .K) ) -}}
{{- $pkgTMatName := ( printf "nativeDenseMatrix%s" (short .K) ) -}}
{{- $pkgTT3Name := ( printf "nativeDenseTensor3%s"  (short .K) ) -}}
{{- $pkgNVecName := ( printf "Vector%s" (short .K) ) -}}
{{- $pkgNMatName := ( printf "Matrix%s" (short .K) ) -}}
{{- $pkgNT3Name := ( printf "Tensor3%s"  (short .K) ) -}}
{{- $vecName := "" -}}
{{- $matName := "" -}}
{{- $T3Name := "" -}}
{{- if .N -}}
	{{- $vecName = $pkgNVecName -}}
	{{- $matName = $pkgNMatName -}}
	{{- $T3Name = $pkgNT3Name -}}
{{- else -}}
	{{- $vecName = $pkgTVecName -}}
	{{- $matName = $pkgTMatName -}}
	{{- $T3Name = $pkgTT3Name -}}
{{end -}}


func Test_{{$vecName}}(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	{{if isRangeable .K -}}
	T = New(WithBacking(Range({{reflectKind .K}}, 0, 6)), WithShape(6))
	{{else -}}
	T = New(Of({{reflectKind .K}}), WithShape(6))
	{{end -}}
	it, err := {{$vecName}}(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_{{$matName}}(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	{{if isRangeable .K -}}
	T = New(WithBacking(Range({{reflectKind .K}}, 0, 6)), WithShape(2, 3))
	{{else -}}
	T = New(Of({{reflectKind .K}}), WithShape(2, 3))
	{{end -}}
	it, err := {{$matName}}(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_{{$T3Name}}(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	{{if isRangeable .K -}}
	T = New(WithBacking(Range({{reflectKind .K}}, 0, 24)), WithShape(2, 3, 4))
	{{else -}}
	T = New(Of({{reflectKind .K}}), WithShape(2, 3, 4))
	{{end -}}
	it, err := {{$T3Name}}(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}
`

var (
	NativeIter      *template.Template
	NativeIterTest  *template.Template
	NativeIterStubs *template.Template
)

func init() {
	NativeIter = template.Must(template.New("NativeIter").Funcs(funcs).Parse(nativeIterRaw))
	NativeIterTest = template.Must(template.New("NativeIterTest").Funcs(funcs).Parse(nativeIterTestRaw))
	NativeIterStubs = template.Must(template.New("NativeStubs").Funcs(funcs).Parse(nativeIterStubsRaw))
}

// generateNativeIterators generates the code for native iterators. `isNative` represents whether the code is generated for the `native` package or not.
// isNative will only  be true for the `purego` build tag.
func generateNativeIterators(isNative bool) func(f io.Writer, ak Kinds) {
	type IterTup struct {
		N bool
		K reflect.Kind
	}
	return func(f io.Writer, ak Kinds) {
		if isNative {
			// checkNativeIteratble is separately generated and placed into util.go in the `native` package
			// so there is no need to generate that here.
			fmt.Fprintf(f, importUnqualifiedTensor)
		} else {
			fmt.Fprintf(f, "%v\n", checkNativeiterable)
		}
		ks := filter(ak.Kinds, isSpecialized)
		for _, k := range ks {
			fmt.Fprintf(f, "/* Native Iterables for %v */\n\n", k)
			NativeIter.Execute(f, IterTup{N: isNative, K: k})
			fmt.Fprint(f, "\n\n")
		}
	}
}

func generateNativeIteratorTests(isNative bool) func(f io.Writer, ak Kinds) {
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
			NativeIterTest.Execute(f, IterTup{N: isNative, K: k})
			fmt.Fprint(f, "\n\n")
		}
	}
}

func generateNativeIteratorStubs(f io.Writer, ak Kinds) {
	fmt.Fprintf(f, importUnsafe) // this is required for go:linkname to work
	ks := filter(ak.Kinds, isSpecialized)
	for _, k := range ks {
		NativeIterStubs.Execute(f, k)
		fmt.Fprint(f, "\n\n")
	}
}

func generateNativeIterChecks(f io.Writer, ak Kinds) {
	fmt.Fprintf(f, importUnqualifiedTensor)
	fmt.Fprintf(f, "%v\n", checkNativeiterable)
}
