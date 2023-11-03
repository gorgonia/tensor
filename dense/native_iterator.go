package dense

import (
	"reflect"
	"unsafe"

	"gorgonia.org/tensor/internal/errors"
)

func checkNativeIterable[DT any](t *Dense[DT], dims int) error {
	if !t.IsNativelyAccessible() {
		return errors.Errorf("%T is not natively accessible", t)
	}

	if t.Dims() != dims {
		return errors.Errorf("%T has a shape of %v. Trying to get an native iterator of %d dims instead", t, t.Shape(), dims)
	}

	if t.F() || t.RequiresIterator() {
		return errors.Errorf("NYI: native iterators for colmajor or unpacked matrices")
	}
	return nil
}

func checkNativeSelectable[DT any](t *Dense[DT], axis int) error {
	if !t.IsNativelyAccessible() {
		return errors.New("Cannot select on non-natively accessible data")
	}
	if axis >= t.Shape().Dims() && !(t.IsScalar() && axis == 0) {
		return errors.Errorf("Cannot select on axis %d. Shape is %v", axis, t.Shape())
	}
	if t.F() || t.RequiresIterator() {
		return errors.Errorf("Not yet implemented: native select for colmajor or unpacked matrices")
	}

	return nil
}

// Vector returns a natively iterable vector.
func Vector[DT any](t *Dense[DT]) (retVal []DT, err error) {
	if err = checkNativeIterable(t, 1); err != nil {
		return nil, err
	}
	return t.Data(), nil
}

func Matrix[DT any](t *Dense[DT]) (retVal [][]DT, err error) {
	if err = checkNativeIterable(t, 2); err != nil {
		return nil, err
	}
	data := t.Data()
	shape := t.Shape()
	strides := t.Strides()

	rows := shape[0]
	cols := shape[1]
	rowStride := strides[0]

	retVal = make([][]DT, rows)
	for i := range retVal {
		start := i * rowStride
		retVal[i] = make([]DT, 0)
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&retVal[i]))
		hdr.Data = uintptr(unsafe.Pointer(&data[start]))
		hdr.Cap = cols
		hdr.Len = cols
	}
	return
}

func Tensor3[DT any](t *Dense[DT]) (retVal [][][]DT, err error) {
	if err = checkNativeIterable(t, 3); err != nil {
		return nil, err
	}
	data := t.Data()
	shape := t.Shape()
	strides := t.Strides()

	layers := shape[0]
	rows := shape[1]
	cols := shape[2]
	layerStride := strides[0]
	rowStride := strides[1]

	retVal = make([][][]DT, layers)
	for i := range retVal {
		retVal[i] = make([][]DT, rows)
		for j := range retVal[i] {
			start := i*layerStride + j*rowStride
			retVal[i][j] = make([]DT, 0)
			hdr := (*reflect.SliceHeader)(unsafe.Pointer(&retVal[i][j]))
			hdr.Data = uintptr(unsafe.Pointer(&data[start]))
			hdr.Cap = cols
			hdr.Len = cols
		}
	}
	return

}

func Tensor4[DT any](t *Dense[DT]) (retVal [][][][]DT, err error) {
	if err = checkNativeIterable(t, 4); err != nil {
		return nil, err
	}
	data := t.Data()
	shape := t.Shape()
	strides := t.Strides()

	blocks := shape[0]
	layers := shape[1]
	rows := shape[2]
	cols := shape[3]
	blockStride := strides[0]
	layerStride := strides[1]
	rowStride := strides[2]

	retVal = make([][][][]DT, blocks)
	for i := range retVal {
		retVal[i] = make([][][]DT, layers)
		for j := range retVal[i] {
			retVal[i][j] = make([][]DT, rows)
			for k := range retVal[i][j] {
				start := i*blockStride + j*layerStride + k*rowStride
				retVal[i][j][k] = make([]DT, 0)
				hdr := (*reflect.SliceHeader)(unsafe.Pointer(&retVal[i][j][k]))
				hdr.Data = uintptr(unsafe.Pointer(&data[start]))
				hdr.Cap = cols
				hdr.Len = cols
			}
		}
	}
	return

}

func Select[DT any](t *Dense[DT], axis int) (retVal [][]DT, err error) {
	panic("NYI")
}
