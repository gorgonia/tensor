package tensor

import (
	"fmt"
	"math"
	"reflect"

	"gorgonia.org/dtype"
)

// Dtype is an alias for dtype.Dtype. This alias is here for backward compatibility purposes, for when users are transitioning out of the older tensor libraries.
type Dtype = dtype.Dtype

// nil type class for skipping type class checks
var nilTC dtype.TypeClass = -1

var parameterizedKinds = [...]reflect.Kind{
	reflect.Array,
	reflect.Chan,
	reflect.Func,
	reflect.Interface,
	reflect.Map,
	reflect.Ptr,
	reflect.Slice,
	reflect.Struct,
}

func isParameterizedKind(k reflect.Kind) bool {
	for _, v := range parameterizedKinds {
		if v == k {
			return true
		}
	}
	return false
}

func isFloat(dt dtype.Dtype) bool { return dt == Float64 || dt == Float32 }

// type aliases
var (
	Bool          = dtype.Bool
	Int           = dtype.Int
	Int8          = dtype.Int8
	Int16         = dtype.Int16
	Int32         = dtype.Int32
	Int64         = dtype.Int64
	Uint          = dtype.Uint
	Uint8         = dtype.Uint8
	Uint16        = dtype.Uint16
	Uint32        = dtype.Uint32
	Uint64        = dtype.Uint64
	Float32       = dtype.Float32
	Float64       = dtype.Float64
	Complex64     = dtype.Complex64
	Complex128    = dtype.Complex128
	String        = dtype.String
	Byte          = dtype.Byte
	Uintptr       = dtype.Uintptr
	UnsafePointer = dtype.UnsafePointer
)

// NormOrder represents the order of the norm. Ideally, we'd only represent norms with a uint/byte.
// But there are norm types that are outside numerical types, such as nuclear norm and fobenius norm.
// So it is internally represented by a float. If Go could use NaN and Inf as consts, it would have been best,
// Instead, we use constructors. Both Nuclear and Frobenius norm types are represented as NaNs
//
// The using of NaN and Inf as "special" Norm types lead to the need for IsInf() and IsFrobenius() and IsNuclear() method
type NormOrder float64

func Norm(ord int) NormOrder   { return NormOrder(float64(ord)) }
func InfNorm() NormOrder       { return NormOrder(math.Inf(1)) }
func NegInfNorm() NormOrder    { return NormOrder(math.Inf(-1)) }
func UnorderedNorm() NormOrder { return NormOrder(math.Float64frombits(0x7ff8000000000001)) }
func FrobeniusNorm() NormOrder { return NormOrder(math.Float64frombits(0x7ff8000000000002)) }
func NuclearNorm() NormOrder   { return NormOrder(math.Float64frombits(0x7ff8000000000003)) }

// Valid() is a helper method that deterines if the norm order is valid. A valid norm order is
// one where the fraction component is 0
func (n NormOrder) Valid() bool {
	switch {
	case math.IsNaN(float64(n)):
		nb := math.Float64bits(float64(n))
		if math.Float64bits(float64(UnorderedNorm())) == nb || math.Float64bits(float64(FrobeniusNorm())) == nb || math.Float64bits(float64(NuclearNorm())) == nb {
			return true
		}
	case math.IsInf(float64(n), 0):
		return true
	default:
		if _, frac := math.Modf(float64(n)); frac == 0.0 {
			return true
		}
	}
	return false
}

// IsUnordered returns true if the NormOrder is not an ordered norm
func (n NormOrder) IsUnordered() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(UnorderedNorm()))
}

// IsFrobenius returns true if the NormOrder is a Frobenius norm
func (n NormOrder) IsFrobenius() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(FrobeniusNorm()))
}

// IsNuclear returns true if the NormOrder is a nuclear norm
func (n NormOrder) IsNuclear() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(NuclearNorm()))
}

func (n NormOrder) IsInf(sign int) bool {
	return math.IsInf(float64(n), sign)
}

func (n NormOrder) String() string {
	switch {
	case n.IsUnordered():
		return "Unordered"
	case n.IsFrobenius():
		return "Frobenius"
	case n.IsNuclear():
		return "Nuclear"
	case n.IsInf(1):
		return "+Inf"
	case n.IsInf(-1):
		return "-Inf"
	default:
		return fmt.Sprintf("Norm %v", float64(n))
	}
	panic("unreachable")
}
