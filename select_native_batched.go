package tensor

import (
	"log"
	"reflect"
	"runtime"
	"unsafe"
)

type BatchedNativeSelectF64 struct {
	t  *Dense
	it [][]float64 // FUTURE: this can be made into generic in the future

	// state

	upper  int // the outer dimension after being "reshaped"
	limit  int // limit as to how many rows the `it` can store
	stride int // stride
	r      int // current row
}

func BatchSelectF64(t *Dense, axis int, limit int) *BatchedNativeSelectF64 {
	if err := checkNativeSelectable(t, axis, Float64); err != nil {
		panic(err)
	}

	if limit <= 0 {
		limit = runtime.NumCPU() // default
	}
	upper := ProdInts(t.Shape()[:axis+1])
	if limit > upper {
		limit = upper
		// `it` should come from nativeSelectF64
	}
	stride := t.Strides()[axis]
	data := t.Float64s()

	it := make([][]float64, 0, limit)
	var i, r int
	for i, r = 0, 0; r < limit; i += stride {
		s := make([]float64, 0)
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&s))
		hdr.Data = uintptr(unsafe.Pointer(&data[i]))
		hdr.Len = stride
		hdr.Cap = stride
		it = append(it, s)
		r++
	}

	return &BatchedNativeSelectF64{
		t:      t,
		it:     it,
		upper:  upper,
		limit:  limit,
		stride: stride,
		r:      r,
	}
}

func (it *BatchedNativeSelectF64) Start() (hasRemainingRows, truncated bool) {
	if it.r != it.limit || len(it.it) != it.limit {
		// then it's been moved, so we reset
		it.Reset()
	}
	hasRemainingRows = it.upper > it.r
	truncated = false
	return
}

// Next moves the next batch into the native iterator.
func (it *BatchedNativeSelectF64) Next() (hasRemaingRows, truncated bool) {
	var (
		i int // data ptr
		r int // relative row
		s int // absolute row
	)
	data := it.t.Float64s()
	for i, r, s = it.r*it.stride, 0, it.r; r < it.limit && s < it.upper; i, r, s = i+it.stride, r+1, s+1 {
		sl := it.it[r]
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&sl))
		hdr.Data = uintptr(unsafe.Pointer(&data[i]))
		hdr.Len = it.stride
		hdr.Cap = it.stride
		it.it[r] = sl
	}
	it.r = s

	log.Printf("r %v limit %v, s %v upper %v", r, it.limit, s, it.upper)

	if r < it.limit {
		// truncate it.it
		it.it = it.it[:r]
		return false, true
	}
	if it.r == it.upper {
		return false, false
	}

	return true, false
}

func (it *BatchedNativeSelectF64) Native() [][]float64 { return it.it }

func (it *BatchedNativeSelectF64) Reset() {
	it.it = it.it[:it.limit]

	data := it.t.Float64s()
	var i, r int
	for i, r = 0, 0; r < it.limit; i += it.stride {
		sl := it.it[r]
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&sl))
		hdr.Data = uintptr(unsafe.Pointer(&data[i]))
		hdr.Len = it.stride
		hdr.Cap = it.stride
		it.it[r] = sl
		r++
	}
}
