package tensor

import (
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
		// this block of code is basically
		// 	it = append(it, data[i:i+stride])
		// TODO: benchmark
		it = append(it, make([]float64, 0))
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&it[len(it)-1]))
		hdr.Data = uintptr(unsafe.Pointer(&data[i]))
		hdr.Len = stride
		hdr.Cap = stride
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

func (it *BatchedNativeSelectF64) Start() (curBatch [][]float64, hasRemainingRows bool) {
	if it.r != it.limit || it.IsTruncated() {
		// then it's been moved, so we reset
		it.Reset()
	}
	curBatch = it.it
	hasRemainingRows = it.upper > it.r
	return
}

// Next moves the next batch into the native iterator.
func (it *BatchedNativeSelectF64) Next() (curBatch [][]float64, hasRemaingRows bool) {
	var (
		i int // data ptr
		r int // relative row / row counter for this batch
		s int // absolute row
	)
	if it.r == it.upper {
		return it.it, false
	}
	data := it.t.Float64s()

	// this loop statement looks scary. But it isn't. Let me break it down:
	// Initialization:
	// 	i := it.r*it.stride // the data pointer is the row number * the stride of the matrix.
	// 	r := 0 		    // loop counter. We're gonna iterate `it.limit` times.
	//	s := it.r 	    // the current row number of the matrix.
	// Condition (continue if the following are true):
	//	r < it.limit 	// we only want to iterate at most `it.limit` times.
	// 	s < it.upper	// we want to make sure we don't iterate more rows than there are rows in the matrix.
	// Next:
	//	i = i + it.stride // we're ready to go to the next row.
	//	r = r+1 	  // we increment the row counter.
	//	s = s+1		  // we increment the absolute row number.
	//
	// Could this be written in a less concise way? Sure. But then there'd be a lot more places to keep track of things.
	for i, r, s = it.r*it.stride, 0, it.r; r < it.limit && s < it.upper; i, r, s = i+it.stride, r+1, s+1 {
		// the block of code below is basically:
		//	it.it[r] = data[i:i+stride]
		//	r++
		// For some reason when this is done, Go actually does a lot more allocations.
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&it.it[r]))
		hdr.Data = uintptr(unsafe.Pointer(&data[i]))
	}
	it.r = s

	if it.r == it.upper && r < it.limit {
		// truncate it.it because iterated rows is less than the limit.
		// This implies that there are some extra rows.
		it.it = it.it[:r]
	}

	return it.it, true
}

func (it *BatchedNativeSelectF64) Native() [][]float64 { return it.it }

func (it *BatchedNativeSelectF64) Reset() {
	it.it = it.it[:it.limit:it.limit]

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
	it.r = r
}

func (it *BatchedNativeSelectF64) IsTruncated() bool { return len(it.it) != it.limit }

type IterSelect struct {
	r      int
	upper  int
	stride int
	total  int
}

func NewIterSelect(t *Dense, axis int) *IterSelect {
	upper := ProdInts(t.Shape()[:axis+1])
	stride := t.Strides()[axis]
	total := t.DataSize()
	return &IterSelect{upper: upper, stride: stride, total: total}
}

func (it *IterSelect) Start() (start, end int, hasRem bool) {
	if it.r > it.stride {
		it.Reset()
	}
	return it.r, it.stride, it.r*it.stride+it.stride < it.total
}

func (it *IterSelect) Next() (start, end int, hasRem bool) {
	it.r += it.stride
	return it.r, it.r + it.stride, it.r+it.stride <= it.total
}

func (it *IterSelect) Reset() { it.r = 0 }
