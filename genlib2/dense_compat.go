package main

import (
	"io"
	"text/template"
)

const importsArrowRaw = `import (
	arrowArray "github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/bitutil"
	arrowTensor "github.com/apache/arrow/go/arrow/tensor"
	arrow "github.com/apache/arrow/go/arrow"
)
`

const conversionsRaw = `func convFromFloat64s(to Dtype, data []float64) interface{} {
	switch to {
	{{range .Kinds -}}
	{{if isNumber . -}}
	case {{reflectKind .}}:
		{{if eq .String "float64" -}}
			retVal := make([]float64, len(data))
			copy(retVal, data)
			return retVal
		{{else if eq .String "float32" -}}
			retVal := make([]float32, len(data))
			for i, v := range data {
				switch {
				case math.IsNaN(v):
					retVal[i] = math32.NaN()
				case math.IsInf(v, 1):
					retVal[i] = math32.Inf(1)
				case math.IsInf(v, -1):
					retVal[i] = math32.Inf(-1)
				default:
					retVal[i] = float32(v)
				}
			}
			return retVal
		{{else if eq .String "complex64" -}}
			retVal := make([]complex64, len(data))
			for i, v := range data {
				switch {
				case math.IsNaN(v):
					retVal[i] = complex64(cmplx.NaN())
				case math.IsInf(v, 0):
					retVal[i] = complex64(cmplx.Inf())
				default:
					retVal[i] = complex(float32(v), float32(0))
				}
			}
			return retVal
		{{else if eq .String "complex128" -}}
			retVal := make([]complex128, len(data))
			for i, v := range data {
				switch {
				case math.IsNaN(v):
					retVal[i] = cmplx.NaN()
				case math.IsInf(v, 0):
					retVal[i] = cmplx.Inf()
				default:
					retVal[i] = complex(v, float64(0))
				}
			}
			return retVal
		{{else -}}
			retVal := make([]{{asType .}}, len(data))
			for i, v :=range data{
				switch {
				case math.IsNaN(v), math.IsInf(v, 0):
					retVal[i] = 0
				default:
					retVal[i] = {{asType .}}(v)
				}
			}
			return retVal
		{{end -}}
	{{end -}}
	{{end -}}
	default:
		panic("Unsupported Dtype")
	}
}

func convToFloat64s(t *Dense) (retVal []float64){
	retVal = make([]float64, t.len())
	switch t.t{
	{{range .Kinds -}}
	{{if isNumber . -}}
	case {{reflectKind .}}:
		{{if eq .String "float64" -}}
			return t.{{sliceOf .}}
		{{else if eq .String "float32" -}}
			for i, v := range t.{{sliceOf .}} {
				switch {
				case math32.IsNaN(v):
					retVal[i] = math.NaN()
				case math32.IsInf(v, 1):
					retVal[i] = math.Inf(1)
				case math32.IsInf(v, -1):
					retVal[i] = math.Inf(-1)
				default:
					retVal[i] = float64(v)
				}
			}
		{{else if eq .String "complex64" -}}
			for i, v := range t.{{sliceOf .}} {
				switch {
				case cmplx.IsNaN(complex128(v)):
					retVal[i] = math.NaN()
				case cmplx.IsInf(complex128(v)):
					retVal[i] = math.Inf(1)
				default:
					retVal[i] = float64(real(v))
				}
			}
		{{else if eq .String "complex128" -}}
			for i, v := range t.{{sliceOf .}} {
				switch {
				case cmplx.IsNaN(v):
					retVal[i] = math.NaN()
				case cmplx.IsInf(v):
					retVal[i] = math.Inf(1)
				default:
					retVal[i] = real(v)
				}
			}
		{{else -}}
			for i, v := range t.{{sliceOf .}} {
				retVal[i]=  float64(v)
			}
		{{end -}}
		return retVal
	{{end -}}
	{{end -}}
	default:
		panic(fmt.Sprintf("Cannot convert *Dense of %v to []float64", t.t))
	}
}

func convToFloat64(x interface{}) float64 {
	switch xt := x.(type) {
	{{range .Kinds -}}
	{{if isNumber . -}}
	case {{asType .}}:
		{{if eq .String "float64 -"}}
			return xt
		{{else if eq .String "complex64" -}}
			return float64(real(xt))
		{{else if eq .String "complex128" -}}
			return real(xt)
		{{else -}}
			return float64(xt)
		{{end -}}
	{{end -}}
	{{end -}}
	default:
		panic("Cannot convert to float64")
	}
}
`

const compatRaw = `// FromMat64 converts a *"gonum/matrix/mat64".Dense into a *tensorf64.Tensor.
func FromMat64(m *mat.Dense, opts ...FuncOpt) *Dense {
	r, c := m.Dims()
	fo := ParseFuncOpts(opts...)
	defer returnOpOpt(fo)
	toCopy := fo.Safe()
	as := fo.As()
	if as.Type == nil {
		as = Float64
	}

	switch as.Kind() {
	{{range .Kinds -}}
	{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		{{if eq .String "float64" -}}
			var backing []float64
			if toCopy {
				backing = make([]float64, len(m.RawMatrix().Data))
				copy(backing, m.RawMatrix().Data)
			} else {
				backing = m.RawMatrix().Data
			}
		{{else -}}
			backing := convFromFloat64s({{asType . | title}}, m.RawMatrix().Data).([]{{asType .}})
		{{end -}}
		retVal := New(WithBacking(backing), WithShape(r, c))
		return retVal
	{{end -}}
	{{end -}}
	default:
		panic(fmt.Sprintf("Unsupported Dtype - cannot convert float64 to %v", as))
	}
	panic("Unreachable")
}


// ToMat64 converts a *Dense to a *mat.Dense. All the values are converted into float64s.
// This function will only convert matrices. Anything *Dense with dimensions larger than 2 will cause an error.
func ToMat64(t *Dense, opts ...FuncOpt) (retVal *mat.Dense, err error) {
	// checks:
	if !t.IsNativelyAccessible() {
		return nil, errors.Errorf("Cannot convert *Dense to *mat.Dense. Data is inaccessible")
	}

	if !t.IsMatrix() {
		// error
		return nil, errors.Errorf("Cannot convert *Dense to *mat.Dense. Expected number of dimensions: <=2, T has got %d dimensions (Shape: %v)", t.Dims(), t.Shape())
	}

	fo := ParseFuncOpts(opts...)
	defer returnOpOpt(fo)
	toCopy := fo.Safe()

	// fix dims
	r := t.Shape()[0]
	c := t.Shape()[1]

	var data []float64
	switch {
	case t.t == Float64 && toCopy  && !t.IsMaterializable():
		data = make([]float64, t.len())
		copy(data, t.Float64s())
	case !t.IsMaterializable():	
		data = convToFloat64s(t)
	default:
		it := newFlatIterator(&t.AP)
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if err = handleNoOp(err); err != nil {
				return
			}
			data = append(data, convToFloat64(t.Get(next)))
		}
		err = nil
		
	}

	retVal = mat.NewDense(r, c, data)
	return
}


`

type ArrowData struct {
	BinaryTypes     []string
	FixedWidthTypes []string
	PrimitiveTypes  []string
}

const compatArrowArrayRaw = `// FromArrowArray converts an "arrow/array".Interface into a Tensor of matching DataType.
func FromArrowArray(a arrowArray.Interface) *Dense {
	a.Retain()
	defer a.Release()

	r := a.Len()

	// TODO(poopoothegorilla): instead of creating bool ValidMask maybe
	// bitmapBytes can be used from arrow API
	mask := make([]bool, r)
	for i := 0; i < r; i++ {
		mask[i] = a.IsNull(i)
	}

	switch a.DataType() {
	{{range .BinaryTypes -}}
	case arrow.BinaryTypes.{{.}}:
		{{if eq . "String" -}}
			backing := make([]string, r)
			for i := 0; i < r; i++ {
				backing[i] = a.(*arrowArray.{{.}}).Value(i)
			}
		{{else -}}
			backing := a.(*arrowArray.{{.}}).{{.}}Values()
		{{end -}}
		retVal := New(WithBacking(backing, mask), WithShape(r, 1))
		return retVal
	{{end -}}
	{{range .FixedWidthTypes -}}
	case arrow.FixedWidthTypes.{{.}}:
		{{if eq . "Boolean" -}}
			backing := make([]bool, r)
			for i := 0; i < r; i++ {
				backing[i] = a.(*arrowArray.{{.}}).Value(i)
			}
		{{else -}}
			backing := a.(*arrowArray.{{.}}).{{.}}Values()
		{{end -}}
		retVal := New(WithBacking(backing, mask), WithShape(r, 1))
		return retVal
	{{end -}}
	{{range .PrimitiveTypes -}}
	case arrow.PrimitiveTypes.{{.}}:
		backing := a.(*arrowArray.{{.}}).{{.}}Values()
		retVal := New(WithBacking(backing, mask), WithShape(r, 1))
		return retVal
	{{end -}}
	default:
		panic(fmt.Sprintf("Unsupported Arrow DataType - %v", a.DataType()))
	}

	panic("Unreachable")
}
`

const compatArrowTensorRaw = `// FromArrowTensor converts an "arrow/tensor".Interface into a Tensor of matching DataType.
func FromArrowTensor(a arrowTensor.Interface) *Dense {
	a.Retain()
	defer a.Release()

	if !a.IsContiguous() {
		panic("Non-contiguous data is Unsupported")
	}

	var shape []int
	for _, val := range a.Shape() {
		shape = append(shape, int(val))
	}

	l := a.Len()
	validMask := a.Data().Buffers()[0].Bytes()
	dataOffset := a.Data().Offset()
	mask := make([]bool, l)
	for i := 0; i < l; i++ {
		mask[i] = len(validMask) != 0 && bitutil.BitIsNotSet(validMask, dataOffset+i)
	}

	switch a.DataType() {
	{{range .PrimitiveTypes -}}
	case arrow.PrimitiveTypes.{{.}}:
		backing := a.(*arrowTensor.{{.}}).{{.}}Values()
		if a.IsColMajor() {
			return New(WithShape(shape...), AsFortran(backing, mask))
		}

		return New(WithShape(shape...), WithBacking(backing, mask))
	{{end -}}
	default:
		panic(fmt.Sprintf("Unsupported Arrow DataType - %v", a.DataType()))
	}

	panic("Unreachable")
}
`

var (
	importsArrow       *template.Template
	conversions        *template.Template
	compats            *template.Template
	compatsArrowArray  *template.Template
	compatsArrowTensor *template.Template
)

func init() {
	importsArrow = template.Must(template.New("imports_arrow").Funcs(funcs).Parse(importsArrowRaw))
	conversions = template.Must(template.New("conversions").Funcs(funcs).Parse(conversionsRaw))
	compats = template.Must(template.New("compat").Funcs(funcs).Parse(compatRaw))
	compatsArrowArray = template.Must(template.New("compat_arrow_array").Funcs(funcs).Parse(compatArrowArrayRaw))
	compatsArrowTensor = template.Must(template.New("compat_arrow_tensor").Funcs(funcs).Parse(compatArrowTensorRaw))
}

func generateDenseCompat(f io.Writer, generic Kinds) {
	// NOTE(poopoothegorilla): an alias is needed for the Arrow Array pkg to prevent naming
	// collisions
	importsArrow.Execute(f, generic)
	conversions.Execute(f, generic)
	compats.Execute(f, generic)
	arrowData := ArrowData{
		BinaryTypes:     arrowBinaryTypes,
		FixedWidthTypes: arrowFixedWidthTypes,
		PrimitiveTypes:  arrowPrimitiveTypes,
	}
	compatsArrowArray.Execute(f, arrowData)
	compatsArrowTensor.Execute(f, arrowData)
}
