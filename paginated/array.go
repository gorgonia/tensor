package paginated

import (
	"unsafe"

	"gorgonia.org/tensor"
)

/*
  This implementation choice may be too slow ...
  See: https://github.com/gorgonia/tensor/blob/master/ALTERNATIVEDESIGNS.md#one-struct-multiple-backing-interfaces
*/

// Array interface can be used to satisfy an object
// which stores the data for a page in-memory.
type Array interface {
	// Apply will apply the function to all values in the array
	Apply(fn interface{}) error
	// At will retrieve that value found at the provided index
	At(int) (interface{}, error)
	// Copy will copy the data from the src array
	Copy(src Array) error
	// Len will return the length of the array
	Len() int
	// SetAt will set the provided value at the specified index
	SetAt(interface{}, int) error
	// SetAll will set all values in the array to the provided value
	SetAll(interface{}) error
	// Slice will return a slice of the array
	Slice(start, end int) (Array, error)
	// Zero will set all values in the array to the zero (default) value of their type
	Zero()
}

// GenArray will generate a new array based on the paginated
// tensor's data type and the page size.
func (p *Tensor) GenArray() Array {
	switch p.Dtype() {
	case tensor.Bool:
		return make(BoolArray, p.pageSize, p.pageSize)
	case tensor.Int:
		return make(IntArray, p.pageSize, p.pageSize)
	case tensor.Int8:
		return make(Int8Array, p.pageSize, p.pageSize)
	case tensor.Int16:
		return make(Int16Array, p.pageSize, p.pageSize)
	case tensor.Int32:
		return make(Int32Array, p.pageSize, p.pageSize)
	case tensor.Int64:
		return make(Int64Array, p.pageSize, p.pageSize)
	case tensor.Uint:
		return make(UIntArray, p.pageSize, p.pageSize)
	case tensor.Uint8, tensor.Byte:
		return make(UInt8Array, p.pageSize, p.pageSize)
	case tensor.Uint16:
		return make(UInt16Array, p.pageSize, p.pageSize)
	case tensor.Uint32:
		return make(UInt32Array, p.pageSize, p.pageSize)
	case tensor.Uint64:
		return make(UInt64Array, p.pageSize, p.pageSize)
	case tensor.Float32:
		return make(Float32Array, p.pageSize, p.pageSize)
	case tensor.Float64:
		return make(Float64Array, p.pageSize, p.pageSize)
	case tensor.Complex64:
		return make(Complex64Array, p.pageSize, p.pageSize)
	case tensor.Complex128:
		return make(Complex128Array, p.pageSize, p.pageSize)
	case tensor.String:
		return make(StringArray, p.pageSize, p.pageSize)
	case tensor.Uintptr:
		return make(UintptrArray, p.pageSize, p.pageSize)
	case tensor.UnsafePointer:
		return make(UnsafeArray, p.pageSize, p.pageSize)
	}

	return nil
}

type BoolArray []bool

func (a BoolArray) Apply(fn interface{}) error {
	f, ok := fn.(func(bool) bool)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a BoolArray) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a BoolArray) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(BoolArray)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a BoolArray) Len() int {
	return len(a)
}

func (a BoolArray) SetAll(v interface{}) error {
	val, ok := v.(bool)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a BoolArray) SetAt(v interface{}, i int) error {
	value, ok := v.(bool)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a BoolArray) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return BoolArray(a[start:end]), nil
}

func (a BoolArray) Zero() {
	for i := range a {
		a[i] = false
	}
}

type IntArray []int

func (a IntArray) Apply(fn interface{}) error {
	f, ok := fn.(func(int) int)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a IntArray) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a IntArray) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(IntArray)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a IntArray) Len() int {
	return len(a)
}

func (a IntArray) SetAll(v interface{}) error {
	val, ok := v.(int)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a IntArray) SetAt(v interface{}, i int) error {
	value, ok := v.(int)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a IntArray) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return IntArray(a[start:end]), nil
}

func (a IntArray) Zero() {
	for i := range a {
		a[i] = 0
	}
}

type Int8Array []int8

func (a Int8Array) Apply(fn interface{}) error {
	f, ok := fn.(func(int8) int8)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a Int8Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a Int8Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(Int8Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a Int8Array) Len() int {
	return len(a)
}

func (a Int8Array) SetAll(v interface{}) error {
	val, ok := v.(int8)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a Int8Array) SetAt(v interface{}, i int) error {
	value, ok := v.(int8)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a Int8Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return Int8Array(a[start:end]), nil
}

func (a Int8Array) Zero() {
	for i := range a {
		a[i] = int8(0)
	}
}

type Int16Array []int16

func (a Int16Array) Apply(fn interface{}) error {
	f, ok := fn.(func(int16) int16)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a Int16Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a Int16Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(Int16Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a Int16Array) Len() int {
	return len(a)
}

func (a Int16Array) SetAll(v interface{}) error {
	val, ok := v.(int16)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a Int16Array) SetAt(v interface{}, i int) error {
	value, ok := v.(int16)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a Int16Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return Int16Array(a[start:end]), nil
}

func (a Int16Array) Zero() {
	for i := range a {
		a[i] = int16(0)
	}
}

type Int32Array []int32

func (a Int32Array) Apply(fn interface{}) error {
	f, ok := fn.(func(int32) int32)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a Int32Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a Int32Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(Int32Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a Int32Array) Len() int {
	return len(a)
}

func (a Int32Array) SetAll(v interface{}) error {
	val, ok := v.(int32)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a Int32Array) SetAt(v interface{}, i int) error {
	value, ok := v.(int32)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a Int32Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return Int32Array(a[start:end]), nil
}

func (a Int32Array) Zero() {
	for i := range a {
		a[i] = int32(0)
	}
}

type Int64Array []int64

func (a Int64Array) Apply(fn interface{}) error {
	f, ok := fn.(func(int64) int64)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a Int64Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a Int64Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(Int64Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a Int64Array) Len() int {
	return len(a)
}

func (a Int64Array) SetAll(v interface{}) error {
	val, ok := v.(int64)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a Int64Array) SetAt(v interface{}, i int) error {
	value, ok := v.(int64)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a Int64Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return Int64Array(a[start:end]), nil
}

func (a Int64Array) Zero() {
	for i := range a {
		a[i] = int64(0)
	}
}

type UIntArray []uint

func (a UIntArray) Apply(fn interface{}) error {
	f, ok := fn.(func(uint) uint)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a UIntArray) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a UIntArray) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(UIntArray)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a UIntArray) Len() int {
	return len(a)
}

func (a UIntArray) SetAll(v interface{}) error {
	val, ok := v.(uint)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a UIntArray) SetAt(v interface{}, i int) error {
	value, ok := v.(uint)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a UIntArray) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return UIntArray(a[start:end]), nil
}

func (a UIntArray) Zero() {
	for i := range a {
		a[i] = uint(0)
	}
}

type UInt8Array []uint8

func (a UInt8Array) Apply(fn interface{}) error {
	f, ok := fn.(func(uint8) uint8)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a UInt8Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a UInt8Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(UInt8Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a UInt8Array) Len() int {
	return len(a)
}

func (a UInt8Array) SetAll(v interface{}) error {
	val, ok := v.(uint8)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a UInt8Array) SetAt(v interface{}, i int) error {
	value, ok := v.(uint8)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a UInt8Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return UInt8Array(a[start:end]), nil
}

func (a UInt8Array) Zero() {
	for i := range a {
		a[i] = uint8(0)
	}
}

type UInt16Array []uint16

func (a UInt16Array) Apply(fn interface{}) error {
	f, ok := fn.(func(uint16) uint16)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a UInt16Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a UInt16Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(UInt16Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a UInt16Array) Len() int {
	return len(a)
}

func (a UInt16Array) SetAll(v interface{}) error {
	val, ok := v.(uint16)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a UInt16Array) SetAt(v interface{}, i int) error {
	value, ok := v.(uint16)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a UInt16Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return UInt16Array(a[start:end]), nil
}

func (a UInt16Array) Zero() {
	for i := range a {
		a[i] = uint16(0)
	}
}

type UInt32Array []uint32

func (a UInt32Array) Apply(fn interface{}) error {
	f, ok := fn.(func(uint32) uint32)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a UInt32Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a UInt32Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(UInt32Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a UInt32Array) Len() int {
	return len(a)
}

func (a UInt32Array) SetAll(v interface{}) error {
	val, ok := v.(uint32)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a UInt32Array) SetAt(v interface{}, i int) error {
	value, ok := v.(uint32)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a UInt32Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return UInt32Array(a[start:end]), nil
}

func (a UInt32Array) Zero() {
	for i := range a {
		a[i] = uint32(0)
	}
}

type UInt64Array []uint64

func (a UInt64Array) Apply(fn interface{}) error {
	f, ok := fn.(func(uint64) uint64)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a UInt64Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a UInt64Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(UInt64Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a UInt64Array) Len() int {
	return len(a)
}

func (a UInt64Array) SetAll(v interface{}) error {
	val, ok := v.(uint64)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a UInt64Array) SetAt(v interface{}, i int) error {
	value, ok := v.(uint64)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a UInt64Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return UInt64Array(a[start:end]), nil
}

func (a UInt64Array) Zero() {
	for i := range a {
		a[i] = uint64(0)
	}
}

type Float32Array []float32

func (a Float32Array) Apply(fn interface{}) error {
	f, ok := fn.(func(float32) float32)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a Float32Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a Float32Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(Float32Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a Float32Array) Len() int {
	return len(a)
}

func (a Float32Array) SetAll(v interface{}) error {
	val, ok := v.(float32)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a Float32Array) SetAt(v interface{}, i int) error {
	value, ok := v.(float32)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a Float32Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return Float32Array(a[start:end]), nil
}

func (a Float32Array) Zero() {
	for i := range a {
		a[i] = float32(0)
	}
}

type Float64Array []float64

func (a Float64Array) Apply(fn interface{}) error {
	f, ok := fn.(func(float64) float64)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a Float64Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a Float64Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(Float64Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a Float64Array) Len() int {
	return len(a)
}

func (a Float64Array) SetAll(v interface{}) error {
	val, ok := v.(float64)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a Float64Array) SetAt(v interface{}, i int) error {
	value, ok := v.(float64)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a Float64Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return Float64Array(a[start:end]), nil
}

func (a Float64Array) Zero() {
	for i := range a {
		a[i] = float64(0)
	}
}

type Complex64Array []complex64

func (a Complex64Array) Apply(fn interface{}) error {
	f, ok := fn.(func(complex64) complex64)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a Complex64Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a Complex64Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(Complex64Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a Complex64Array) Len() int {
	return len(a)
}

func (a Complex64Array) SetAll(v interface{}) error {
	val, ok := v.(complex64)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a Complex64Array) SetAt(v interface{}, i int) error {
	value, ok := v.(complex64)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a Complex64Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return Complex64Array(a[start:end]), nil
}

func (a Complex64Array) Zero() {
	for i := range a {
		a[i] = complex64(0)
	}
}

type Complex128Array []complex128

func (a Complex128Array) Apply(fn interface{}) error {
	f, ok := fn.(func(complex128) complex128)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a Complex128Array) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a Complex128Array) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(Complex128Array)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a Complex128Array) Len() int {
	return len(a)
}

func (a Complex128Array) SetAll(v interface{}) error {
	val, ok := v.(complex128)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a Complex128Array) SetAt(v interface{}, i int) error {
	value, ok := v.(complex128)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a Complex128Array) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return Complex128Array(a[start:end]), nil
}

func (a Complex128Array) Zero() {
	for i := range a {
		a[i] = complex128(0)
	}
}

type StringArray []string

func (a StringArray) Apply(fn interface{}) error {
	f, ok := fn.(func(string) string)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a StringArray) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a StringArray) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(StringArray)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a StringArray) Len() int {
	return len(a)
}

func (a StringArray) SetAll(v interface{}) error {
	val, ok := v.(string)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a StringArray) SetAt(v interface{}, i int) error {
	value, ok := v.(string)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a StringArray) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return StringArray(a[start:end]), nil
}

func (a StringArray) Zero() {
	for i := range a {
		a[i] = ""
	}
}

type UintptrArray []uintptr

func (a UintptrArray) Apply(fn interface{}) error {
	f, ok := fn.(func(uintptr) uintptr)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a UintptrArray) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a UintptrArray) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(UintptrArray)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a UintptrArray) Len() int {
	return len(a)
}

func (a UintptrArray) SetAll(v interface{}) error {
	val, ok := v.(uintptr)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a UintptrArray) SetAt(v interface{}, i int) error {
	value, ok := v.(uintptr)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a UintptrArray) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return UintptrArray(a[start:end]), nil
}

func (a UintptrArray) Zero() {
	for i := range a {
		a[i] = uintptr(0)
	}
}

type UnsafeArray []unsafe.Pointer

func (a UnsafeArray) Apply(fn interface{}) error {
	f, ok := fn.(func(unsafe.Pointer) unsafe.Pointer)
	if !ok {
		return ErrFuncType
	}

	for i, v := range a {
		a[i] = f(v)
	}

	return nil
}

func (a UnsafeArray) At(i int) (interface{}, error) {
	if i > 0 && i < len(a) {
		return a[i], nil
	}
	return nil, ErrBound
}

func (a UnsafeArray) Copy(src Array) error {
	if len(a) < src.Len() {
		return ErrLength
	}

	array, ok := src.(UnsafeArray)
	if !ok {
		return ErrType
	}

	for i, v := range array {
		a[i] = v
	}

	return nil
}

func (a UnsafeArray) Len() int {
	return len(a)
}

func (a UnsafeArray) SetAll(v interface{}) error {
	val, ok := v.(unsafe.Pointer)
	if !ok {
		return ErrType
	}

	for i := range a {
		a[i] = val
	}

	return nil
}

func (a UnsafeArray) SetAt(v interface{}, i int) error {
	value, ok := v.(unsafe.Pointer)
	if ok {
		if i > 0 && i < len(a) {
			a[i] = value
			return nil
		}
	}

	return ErrType
}

func (a UnsafeArray) Slice(start, end int) (Array, error) {
	if start < 0 || end > len(a) || start > len(a) || end < start {
		return nil, ErrBound
	}

	return UnsafeArray(a[start:end]), nil
}

func (a UnsafeArray) Zero() {
	for i := range a {
		a[i] = nil
	}
}
