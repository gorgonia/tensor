package serialization

// proteus:generate
type DataOrder byte

// the reason for spreading the states out is because proteaus cannot handle non-iota tates
const (
	RowMajorContiguous = iota
	RowMajorNonContiguous
	ColMajorContiguous
	ColMajorNonContiguous
)

type Triangle byte

const (
	NotTriangle Triangle = iota
	Upper
	Lower
	Symmetric
)

// proteus:generate
type AP struct {
	Shape   []int
	Strides []int

	O DataOrder
	T Triangle
}

// proteus:generate
type Dense struct {
	AP
	Type string // type name
	Data []byte
}

// proteus:generate
type MaskedDense struct {
	Dense
	Mask       []bool
	MaskIsSoft []bool
}
