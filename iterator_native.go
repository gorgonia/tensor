package tensor

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

func checkNativeable(t *Dense, dims int, dt Dtype) error {
	// checks:
	if !t.IsNativelyAccessible() {
		return errors.Errorf("Cannot convert *Dense to *mat.Dense. Data is inaccessible")
	}

	if t.shape.Dims() != dims {
		return errors.Errorf("Cannot convert *Dense to native iterator. Expected number of dimension: %d, T has got %d dimensions (Shape: %v)", dims, t.Dims(), t.Shape())
	}

	if t.DataOrder().isColMajor() || t.RequiresIterator() {
		return errors.Errorf("Not yet implemented: native matrix for colmajor or unpacked matrices")
	}

	if t.Dtype() != dt {
		return errors.Errorf("NativeMatrixF64 only works for float64")
	}

	return nil
}

func NativeVectorF64(t *Dense) (retVal []float64, err error) {
	if err = checkNativeable(t, 1, Float64); err != nil {
		return nil, err
	}
	return t.Data().([]float64), nil
}

func NativeMatrixF64(t *Dense) (retVal [][]float64, err error) {
	if err = checkNativeable(t, 2, Float64); err != nil {
		return nil, err
	}

	data := t.Data().([]float64)
	shape := t.Shape()
	strides := t.Strides()

	rows := shape[0]
	cols := shape[1]
	rowStride := strides[0]
	retVal = make([][]float64, rows)
	for i := range retVal {
		start := i * rowStride
		hdr := &reflect.SliceHeader{
			Data: uintptr(unsafe.Pointer(&data[start])),
			Len:  cols,
			Cap:  cols,
		}
		retVal[i] = *(*[]float64)(unsafe.Pointer(hdr))
	}
	return
}

func Native3TensorF64(t *Dense) (retVal [][][]float64, err error) {
	if err = checkNativeable(t, 3, Float64); err != nil {
		return nil, err
	}

	data := t.Data().([]float64)
	shape := t.Shape()
	strides := t.Strides()

	layers := shape[0]
	rows := shape[1]
	cols := shape[2]
	layerStride := strides[0]
	rowStride := strides[1]
	retVal = make([][][]float64, layers)
	for i := range retVal {
		retVal[i] = make([][]float64, rows)
		for j := range retVal[i] {
			start := i*layerStride + j*rowStride
			hdr := &reflect.SliceHeader{
				Data: uintptr(unsafe.Pointer(&data[start])),
				Len:  cols,
				Cap:  cols,
			}
			retVal[i][j] = *(*[]float64)(unsafe.Pointer(hdr))
		}
	}
	return
}
