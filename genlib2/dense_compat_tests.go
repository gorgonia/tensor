package main

import (
	"io"
	"text/template"
)

const compatTestsRaw = `var toMat64Tests = []struct{
	data interface{}
	sliced interface{}
	shape Shape
	dt Dtype
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	{ Range({{asType . | title | strip}}, 0, 6), []{{asType .}}{0,1,3,4}, Shape{2,3}, {{asType . | title | strip}} },
	{{end -}}
	{{end -}}
}
func TestToMat64(t *testing.T){
	assert := assert.New(t)
	for i, tmt := range toMat64Tests {
		T := New(WithBacking(tmt.data), WithShape(tmt.shape...))
		var m *mat.Dense
		var err error
		if m, err = ToMat64(T); err != nil {
			t.Errorf("ToMat basic test %d failed : %v", i, err)
			continue
		}
		conv := anyToFloat64s(tmt.data)
		assert.Equal(conv, m.RawMatrix().Data, "i %d from %v", i, tmt.dt)

		if T, err = sliceDense(T, nil, makeRS(0, 2)); err != nil{
			t.Errorf("Slice failed %v", err)
			continue
		}
		if m, err = ToMat64(T); err != nil {
			t.Errorf("ToMat of slice test %d failed : %v", i, err)
			continue
		}
		conv = anyToFloat64s(tmt.sliced)
		assert.Equal(conv, m.RawMatrix().Data, "sliced test %d from %v", i, tmt.dt)
		t.Logf("Done")

		if tmt.dt == Float64 {
			T = New(WithBacking(tmt.data), WithShape(tmt.shape...))
			if m, err = ToMat64(T, UseUnsafe()); err != nil {
				t.Errorf("ToMat64 unsafe test %d failed: %v", i, err)
			}
			conv = anyToFloat64s(tmt.data)
			assert.Equal(conv, m.RawMatrix().Data, "float64 unsafe i %d from %v", i, tmt.dt)
			conv[0] = 1000
			assert.Equal(conv, m.RawMatrix().Data,"float64 unsafe i %d from %v", i, tmt.dt)
			conv[0] = 0 // reset for future tests that use the same backing
		}
	}
	// idiocy test
	T := New(Of(Float64), WithShape(2,3,4))
	_, err := ToMat64(T)
	if err == nil {
		t.Error("Expected an error when trying to convert a 3-T to *mat.Dense")
	}
}

func TestFromMat64(t *testing.T){
	assert := assert.New(t)
	var m *mat.Dense
	var T *Dense
	var backing []float64


	for i, tmt := range toMat64Tests {
		backing = Range(Float64, 0, 6).([]float64)
		m = mat.NewDense(2, 3, backing)
		T = FromMat64(m)
		conv := anyToFloat64s(tmt.data)
		assert.Equal(conv, T.Float64s(), "test %d: []float64 from %v", i, tmt.dt)
		assert.True(T.Shape().Eq(tmt.shape))

		T = FromMat64(m, As(tmt.dt))
		assert.Equal(tmt.data, T.Data())
		assert.True(T.Shape().Eq(tmt.shape))

		if tmt.dt == Float64{
			backing = Range(Float64, 0, 6).([]float64)
			m = mat.NewDense(2, 3, backing)
			T = FromMat64(m, UseUnsafe())
			assert.Equal(backing, T.Float64s())
			assert.True(T.Shape().Eq(tmt.shape))
			backing[0] = 1000 
			assert.Equal(backing, T.Float64s(), "test %d - unsafe float64", i)
		}
	}
}
`

const compatArrowArrayTestsRaw = `var toArrowArrayTests = []struct{
	data interface{}
	valid []bool
	dt arrow.DataType
	shape Shape
}{
	{{range .PrimitiveTypes -}}
	{
		data: Range({{.}}, 0, 6),
		valid: []bool{true, true, true, false, true, true},
		dt: arrow.PrimitiveTypes.{{ . }},
		shape: Shape{6,1},
	},
	{{end -}}
}
func TestFromArrowArray(t *testing.T){
	assert := assert.New(t)
	var T *Dense
	pool := memory.NewGoAllocator()

	for i, taat := range toArrowArrayTests {
		var m arrowArray.Interface

		switch taat.dt {
		{{range .BinaryTypes -}}
		case arrow.BinaryTypes.{{ . }}:
			b := arrowArray.New{{ . }}Builder(pool)
			defer b.Release()
			b.AppendValues(
				{{if eq . "String" -}}
				[]string{"0", "1", "2", "3", "4", "5"},
				{{else -}}
				Range({{ . }}, 0, 6).([]{{lower . }}),
				{{end -}}
				taat.valid,
			)
			m = b.NewArray()
			defer m.Release()
		{{end -}}
		{{range .FixedWidthTypes -}}
		case arrow.FixedWidthTypes.{{ . }}:
			b := arrowArray.New{{ . }}Builder(pool)
			defer b.Release()
			b.AppendValues(
				{{if eq . "Boolean" -}}
				[]bool{true, false, true, false, true, false},
				{{else -}}
				Range({{ . }}, 0, 6).([]{{lower . }}),
				{{end -}}
				taat.valid,
			)
			m = b.NewArray()
			defer m.Release()
		{{end -}}
		{{range .PrimitiveTypes -}}
		case arrow.PrimitiveTypes.{{ . }}:
			b := arrowArray.New{{ . }}Builder(pool)
			defer b.Release()
			b.AppendValues(
				Range({{ . }}, 0, 6).([]{{lower . }}),
				taat.valid,
			)
			m = b.NewArray()
			defer m.Release()
		{{end -}}
		default:
			t.Errorf("DataType not supported in tests: %v", taat.dt)
		}

		T = FromArrowArray(m)
		switch taat.dt {
		{{range .PrimitiveTypes -}}
		case arrow.PrimitiveTypes.{{ . }}:
			conv := taat.data.([]{{lower . }})
			assert.Equal(conv, T.{{ . }}s(), "test %d: []{{lower . }} from %v", i, taat.dt)
		{{end -}}
		default:
			t.Errorf("DataType not supported in tests: %v", taat.dt)
		}
		for i, invalid := range T.Mask() {
			assert.Equal(taat.valid[i], !invalid)
		}
		assert.True(T.Shape().Eq(taat.shape))
	}
}
`

const compatArrowTensorTestsRaw = `var toArrowTensorTests = []struct{
	rowMajorData interface{}
	colMajorData interface{}
	rowMajorValid []bool
	colMajorValid []bool
	dt arrow.DataType
	shape Shape
}{
	{{range .PrimitiveTypes -}}
	{
		rowMajorData: []{{lower .}}{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		colMajorData: []{{lower .}}{1, 6, 2, 7, 3, 8, 4, 9, 5, 10},
		rowMajorValid: []bool{true, false, true, false, true, false, true, false, true, false},
		colMajorValid: []bool{true, false, false, true, true, false, false, true, true, false},
		dt: arrow.PrimitiveTypes.{{ . }},
		shape: Shape{2,5},
	},
	{{end -}}
}
func TestFromArrowTensor(t *testing.T){
	assert := assert.New(t)
	var rowMajorT *Dense
	var colMajorT *Dense
	pool := memory.NewGoAllocator()

	for i, taat := range toArrowTensorTests {
		var rowMajorArr arrowArray.Interface
		var colMajorArr arrowArray.Interface
		var rowMajor arrowTensor.Interface
		var colMajor arrowTensor.Interface

		switch taat.dt {
		{{range .PrimitiveTypes -}}
		case arrow.PrimitiveTypes.{{ . }}:
			b := arrowArray.New{{ . }}Builder(pool)
			defer b.Release()
			b.AppendValues(
				[]{{lower . }}{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
				taat.rowMajorValid,
			)
			rowMajorArr = b.NewArray()
			defer rowMajorArr.Release()

			b.AppendValues(
				[]{{lower .}}{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
				taat.rowMajorValid,
			)
			colMajorArr = b.NewArray()
			defer colMajorArr.Release()

			rowMajor = arrowTensor.New{{.}}(rowMajorArr.Data(), []int64{2, 5}, nil, []string{"x", "y"})
			defer rowMajor.Release()
			colMajor = arrowTensor.New{{.}}(colMajorArr.Data(), []int64{2, 5}, []int64{int64(arrow.{{ . }}SizeBytes), int64(arrow.{{ . }}SizeBytes * 2)}, []string{"x", "y"})
			defer colMajor.Release()
		{{end -}}
		default:
			t.Errorf("DataType not supported in tests: %v", taat.dt)
		}

		rowMajorT = FromArrowTensor(rowMajor)
		colMajorT = FromArrowTensor(colMajor)

		assert.Equal(taat.rowMajorData, rowMajorT.Data(), "test %d: row major %v", i, taat.dt)
		assert.Equal(len(taat.rowMajorValid), len(rowMajorT.Mask()), "test %d: row major %v mask length incorrect", i, taat.dt)
		for i, invalid := range rowMajorT.Mask() {
			assert.Equal(taat.rowMajorValid[i], !invalid, "test %d: row major %v mask value incorrect", i, taat.dt)
		}
		assert.True(colMajorT.Shape().Eq(taat.shape))

		assert.Equal(taat.colMajorData, colMajorT.Data(), "test %d: column major %v", i, taat.dt)
		assert.Equal(len(taat.colMajorValid), len(colMajorT.Mask()), "test %d: column major %v mask length incorrect", i, taat.dt)
		for i, invalid := range colMajorT.Mask() {
			assert.Equal(taat.colMajorValid[i], !invalid, "test %d: column major %v mask value incorrect", i, taat.dt)
		}
		assert.True(rowMajorT.Shape().Eq(taat.shape))
	}
}
`

var (
	compatTests            *template.Template
	compatArrowArrayTests  *template.Template
	compatArrowTensorTests *template.Template
)

func init() {
	compatTests = template.Must(template.New("testCompat").Funcs(funcs).Parse(compatTestsRaw))
	compatArrowArrayTests = template.Must(template.New("testArrowArrayCompat").Funcs(funcs).Parse(compatArrowArrayTestsRaw))
	compatArrowTensorTests = template.Must(template.New("testArrowTensorCompat").Funcs(funcs).Parse(compatArrowTensorTestsRaw))
}

func generateDenseCompatTests(f io.Writer, generic Kinds) {
	// NOTE(poopoothegorilla): an alias is needed for the Arrow Array pkg to prevent naming
	// collisions
	importsArrow.Execute(f, generic)
	compatTests.Execute(f, generic)
	arrowData := ArrowData{
		BinaryTypes:     arrowBinaryTypes,
		FixedWidthTypes: arrowFixedWidthTypes,
		PrimitiveTypes:  arrowPrimitiveTypes,
	}
	compatArrowArrayTests.Execute(f, arrowData)
	compatArrowTensorTests.Execute(f, arrowData)
}
