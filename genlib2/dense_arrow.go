package main

type ArrowData struct {
	BinaryTypes     []string
	FixedWidthTypes []string
	PrimitiveTypes  []string
}

const compatArrowRaw = `// FromArrowArray converts an "arrow/array".Interface into a Tensor of matching DataType.
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
	{{range .ArrowData.PrimitiveTypes -}}
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
