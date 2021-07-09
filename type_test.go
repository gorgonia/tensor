package tensor

import (
	"gorgonia.org/dtype"
)

var numberTypes = []dtype.Dtype{
	Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Float32, Float64, Complex64, Complex128,
}

var specializedTypes = []dtype.Dtype{
	Bool, Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Float32, Float64, Complex64, Complex128, String,
}
