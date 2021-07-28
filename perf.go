package tensor

import (
	"runtime"
	"sync"

	"gorgonia.org/dtype"
	"gorgonia.org/tensor/internal/storage"
)

var habbo sync.Mutex
var usePool = true

// tensorPool is a pool of *Tensor grouped by size. It's guarded by poolsClosed

const (
	maxAPDims = 8
	maxDims   = 8
	PoolSize  = 4096
)

// UsePool enables the use of a pool of *Tensors as provided in the package. This is the default option
func UsePool() {
	habbo.Lock()
	usePool = true
	habbo.Unlock()
}

// DontUsePool makes sure the functions don't use the tensor pool provided.
// This is useful as certain applications don't lend themselves well to use of the pool.
// Examples of such applications would be one where many tensors of wildly different sizes are created all the time.
func DontUsePool() {
	habbo.Lock()
	usePool = false
	habbo.Unlock()
}

// headerPool should ever only be used by scalarToHeader
var headerPool = make(chan *storage.Header, PoolSize)

func borrowHeader() *storage.Header {
	select {
	case hdr := <-headerPool:
		return hdr
	default:
		hdr := new(storage.Header)
		runtime.SetFinalizer(hdr, destroyHeader)
		return hdr
	}
}

func returnHeader(hdr *storage.Header) {
	destroyHeader(hdr)
	if len(headerPool) < cap(headerPool) {
		headerPool <- hdr
	}
}

func destroyHeader(hdr *storage.Header) {
	hdr.Raw = nil
}

var densePool = make(chan *Dense, PoolSize)

func borrowDense() *Dense {
	select {
	case t := <-densePool:
		return t
	default:
		t := new(Dense)
		t.e = StdEng{}
		// t.oe = StdEng{}
		return t
	}
	// return densePool.Get().(*Dense)
}

// ReturnTensor returns a Tensor to their respective pools. USE WITH CAUTION
func ReturnTensor(t Tensor) {
	if !usePool {
		return
	}
	switch tt := t.(type) {
	case *Dense:
		tt.AP.zero()

		if tt.transposeWith != nil {
			ReturnInts(tt.transposeWith)
			tt.transposeWith = nil
		}

		// array reset
		tt.t = dtype.Dtype{}
		tt.array.Header.Raw = nil

		// engine and flag reset
		tt.e = StdEng{}
		tt.oe = nil
		tt.flag = 0

		// other reset
		tt.old.zero()
		tt.viewOf = 0
		tt.transposeWith = nil

		// mask related stuff - TODO: deprecate
		tt.mask = nil
		tt.maskIsSoft = false

		// densePool.Put(tt)
		if len(densePool) < cap(densePool) {
			densePool <- tt
		}
	}
}

/* ----------------------------------------------------------------
------------------ Create Pools
------------------------------------------------------------------*/

/* APLIST POOL */

// Init function
func init() {

	for i := range intsPool {
		size := i
		intsPool[i].New = func() interface{} { return make([]int, size) }
	}

	// for i := range boolsPool {
	// 	size := i
	// 	boolsPool[i].New = func() interface{} { return make([]bool, size) }
	// }
}

/* INTS POOL */

var intsPool [maxDims + 1]sync.Pool

// var intsPool = make(chan []int, PoolSize)

/* BOOLS POOL */
var boolsPool = make(chan []bool, PoolSize)

// var boolsPool [PoolSize]sync.Pool

// BorrowInts borrows a slice of ints from the pool. USE WITH CAUTION.
func BorrowInts(size int) []int {
	if size > maxDims {
		return make([]int, size, size)
	}

	// select {
	// case ints := <-intsPool:
	// 	ints = ints[:size]
	// 	return ints
	// default:
	// 	ints := make([]int, size, 8)
	// 	return ints
	// }
	retVal := intsPool[size].Get()
	if retVal == nil {
		return make([]int, size)
	}
	// log.Printf("Borrowing %p. Called by %v", retVal, string(debug.Stack()))
	return retVal.([]int)[:size]
}

// ReturnInts returns a slice from the pool. USE WITH CAUTION.
func ReturnInts(is []int) {
	// log.Printf("Returning %p. Called by %v", is, string(debug.Stack()))
	if is == nil {
		return
	}
	// if len(is) == 2 && is[0] == 52 && is[1] == 10 {
	// 	log.Printf("ints %v", is)
	// 	pc, _, _, _ := runtime.Caller(3)
	// 	log.Printf("Called: %v", runtime.FuncForPC(pc).Name())
	// }
	size := cap(is)
	if size > maxDims {
		return
	}
	is = is[:cap(is)]
	for i := range is {
		is[i] = 0
	}

	// if len(intsPool) < cap(intsPool) {
	// 	intsPool <- is
	// }

	intsPool[size].Put(is)
}

// BorrowBools borrows a slice of bools from the pool. USE WITH CAUTION.
func BorrowBools(size int) []bool {
	if size >= 8 {
		return make([]bool, size)
	}

	select {
	case bools := <-boolsPool:
		return bools
	default:
		bools := make([]bool, 8)
		bools = bools[:size]
		return bools
	}

	// retVal := boolsPool[size].Get()
	// if retVal == nil {
	// 	return make([]bool, size)
	// }
	// return retVal.([]bool)
}

// ReturnBools returns a slice from the pool. USE WITH CAUTION.
func ReturnBools(is []bool) {
	if is == nil {
		return
	}
	size := cap(is)
	if size >= 8 {
		return
	}
	is = is[:cap(is)]
	for i := range is {
		is[i] = false
	}

	if len(boolsPool) < cap(boolsPool) {
		boolsPool <- is
	}
	// boolsPool[size].Put(is)
}

// var optPool = make(chan *OpOpt, PoolSize)
// var optPool = newRingbuffer(PoolSize)
var optPool = &sync.Pool{
	New: func() interface{} { return new(opOpt) },
}

func borrowOpOpt() *opOpt {
	// select {
	// case fo := <-optPool:
	// 	return fo
	// default:
	// 	return new(OpOpt)
	// }

	return optPool.Get().(*opOpt)

	// if fo, err := optPool.Get(); err == nil {
	// 	return (*OpOpt)(fo)
	// }
	// return new(OpOpt)
}

func returnOpOpt(oo *opOpt) {
	oo.reuse = nil
	oo.incr = nil
	oo.unsafe = false
	oo.same = false
	oo.t = dtype.Dtype{}
	oo.ctx = nil
	// if len(optPool) < cap(optPool) {
	// 	optPool <- oo
	// }

	optPool.Put(oo)

	// optPool.Put(unsafe.Pointer(oo))
}
