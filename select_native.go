package tensor

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/dtype"
)

// Code generated by genlib2. DO NOT EDIT.

func checkNativeSelectable(t *Dense, axis int, dt dtype.Dtype) error {
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

/* Native Select for bool */

// nativeSelectB creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectB(t *Dense, axis int) (retVal [][]bool, err error) {
	if err := checkNativeSelectable(t, axis, Bool); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]bool, 1)
		retVal[0] = t.Bools()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixB(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Bools()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]bool, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]bool, 0)
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

/* Native Select for int */

// nativeSelectI creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectI(t *Dense, axis int) (retVal [][]int, err error) {
	if err := checkNativeSelectable(t, axis, Int); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]int, 1)
		retVal[0] = t.Ints()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixI(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Ints()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]int, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]int, 0)
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

/* Native Select for int8 */

// nativeSelectI8 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectI8(t *Dense, axis int) (retVal [][]int8, err error) {
	if err := checkNativeSelectable(t, axis, Int8); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]int8, 1)
		retVal[0] = t.Int8s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixI8(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Int8s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]int8, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]int8, 0)
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

/* Native Select for int16 */

// nativeSelectI16 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectI16(t *Dense, axis int) (retVal [][]int16, err error) {
	if err := checkNativeSelectable(t, axis, Int16); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]int16, 1)
		retVal[0] = t.Int16s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixI16(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Int16s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]int16, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]int16, 0)
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

/* Native Select for int32 */

// nativeSelectI32 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectI32(t *Dense, axis int) (retVal [][]int32, err error) {
	if err := checkNativeSelectable(t, axis, Int32); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]int32, 1)
		retVal[0] = t.Int32s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixI32(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Int32s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]int32, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]int32, 0)
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

/* Native Select for int64 */

// nativeSelectI64 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectI64(t *Dense, axis int) (retVal [][]int64, err error) {
	if err := checkNativeSelectable(t, axis, Int64); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]int64, 1)
		retVal[0] = t.Int64s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixI64(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Int64s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]int64, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]int64, 0)
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

/* Native Select for uint */

// nativeSelectU creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectU(t *Dense, axis int) (retVal [][]uint, err error) {
	if err := checkNativeSelectable(t, axis, Uint); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]uint, 1)
		retVal[0] = t.Uints()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixU(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Uints()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]uint, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]uint, 0)
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

/* Native Select for uint8 */

// nativeSelectU8 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectU8(t *Dense, axis int) (retVal [][]uint8, err error) {
	if err := checkNativeSelectable(t, axis, Uint8); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]uint8, 1)
		retVal[0] = t.Uint8s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixU8(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Uint8s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]uint8, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]uint8, 0)
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

/* Native Select for uint16 */

// nativeSelectU16 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectU16(t *Dense, axis int) (retVal [][]uint16, err error) {
	if err := checkNativeSelectable(t, axis, Uint16); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]uint16, 1)
		retVal[0] = t.Uint16s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixU16(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Uint16s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]uint16, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]uint16, 0)
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

/* Native Select for uint32 */

// nativeSelectU32 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectU32(t *Dense, axis int) (retVal [][]uint32, err error) {
	if err := checkNativeSelectable(t, axis, Uint32); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]uint32, 1)
		retVal[0] = t.Uint32s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixU32(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Uint32s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]uint32, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]uint32, 0)
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

/* Native Select for uint64 */

// nativeSelectU64 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectU64(t *Dense, axis int) (retVal [][]uint64, err error) {
	if err := checkNativeSelectable(t, axis, Uint64); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]uint64, 1)
		retVal[0] = t.Uint64s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixU64(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Uint64s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]uint64, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]uint64, 0)
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

/* Native Select for float32 */

// nativeSelectF32 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectF32(t *Dense, axis int) (retVal [][]float32, err error) {
	if err := checkNativeSelectable(t, axis, Float32); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]float32, 1)
		retVal[0] = t.Float32s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixF32(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Float32s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]float32, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]float32, 0)
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

/* Native Select for float64 */

// nativeSelectF64 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectF64(t *Dense, axis int) (retVal [][]float64, err error) {
	if err := checkNativeSelectable(t, axis, Float64); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]float64, 1)
		retVal[0] = t.Float64s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixF64(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Float64s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]float64, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]float64, 0)
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

/* Native Select for complex64 */

// nativeSelectC64 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectC64(t *Dense, axis int) (retVal [][]complex64, err error) {
	if err := checkNativeSelectable(t, axis, Complex64); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]complex64, 1)
		retVal[0] = t.Complex64s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixC64(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Complex64s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]complex64, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]complex64, 0)
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

/* Native Select for complex128 */

// nativeSelectC128 creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectC128(t *Dense, axis int) (retVal [][]complex128, err error) {
	if err := checkNativeSelectable(t, axis, Complex128); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]complex128, 1)
		retVal[0] = t.Complex128s()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixC128(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Complex128s()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]complex128, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]complex128, 0)
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

/* Native Select for string */

// nativeSelectStr creates a slice of flat data types. See Example of NativeSelectF64.
func nativeSelectStr(t *Dense, axis int) (retVal [][]string, err error) {
	if err := checkNativeSelectable(t, axis, String); err != nil {
		return nil, err
	}

	switch t.Shape().Dims() {
	case 0, 1:
		retVal = make([][]string, 1)
		retVal[0] = t.Strings()
	case 2:
		if axis == 0 {
			return nativeDenseMatrixStr(t)
		}
		fallthrough
	default:
		// size := t.Shape()[axis]
		data := t.Strings()
		stride := t.Strides()[axis]
		upper := ProdInts(t.Shape()[:axis+1])
		retVal = make([][]string, 0, upper)
		for i, r := 0, 0; r < upper; i += stride {
			s := make([]string, 0)
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
