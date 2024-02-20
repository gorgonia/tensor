package sparse

import (
	"sort"

	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
)

// coo is an internal representation of the Coordinate type sparse matrix.
// It's not exported because you probably shouldn't be using it.
// Instead, constructors for the *CS type supports using a coordinate as an input.
type coo[DT any] struct {
	o      DataOrder
	xs, ys []int
	data   []DT
	bytes  []byte
}

func (c *coo[DT]) Len() int { return len(c.data) }
func (c *coo[DT]) Less(i, j int) bool {
	if c.o.IsColMajor() {
		return c.colMajorLess(i, j)
	}
	return c.rowMajorLess(i, j)
}
func (c *coo[DT]) Swap(i, j int) {
	c.xs[i], c.xs[j] = c.xs[j], c.xs[i]
	c.ys[i], c.ys[j] = c.ys[j], c.ys[i]
	c.data[i], c.data[j] = c.data[j], c.data[i]
}

func (c *coo[DT]) colMajorLess(i, j int) bool {
	if c.ys[i] < c.ys[j] {
		return true
	}
	if c.ys[i] == c.ys[j] {
		// check xs
		if c.xs[i] <= c.xs[j] {
			return true
		}
	}
	return false
}

func (c *coo[DT]) rowMajorLess(i, j int) bool {
	if c.xs[i] < c.xs[j] {
		return true
	}

	if c.xs[i] == c.xs[j] {
		// check ys
		if c.ys[i] <= c.ys[j] {
			return true
		}
	}
	return false
}

// CS is a compressed sparse data structure. It can be used to represent both CSC and CSR sparse matrices.
// Refer to the individual creation functions for more information.
type CS[DT any] struct {
	s shapes.Shape
	o DataOrder
	e Engine
	f MemoryFlag
	t dtype.Dtype
	z DT // z is the "zero" value. Typically it's not used.

	indices []int
	indptr  []int

	data  []DT
	bytes []byte
}

// NewCSR creates a new Compressed Sparse Row matrix. The data has to be a slice or it panics.
func NewCSR[DT any](indices, indptr []int, data []DT, opts ...ConsOpt) *CS[DT] {
	t := new(CS)
	t.indices = indices
	t.indptr = indptr
	t.array = arrayFromSlice(data)
	t.o = NonContiguous
	t.e = StdEng{}

	for _, opt := range opts {
		opt(t)
	}
	return t
}

// NewCSC creates a new Compressed Sparse Column matrix. The data has to be a slice, or it panics.
func NewCSC[DT any](indices, indptr []int, data []DT, opts ...ConsOpt) *CS[DT] {
	t := new(CS)
	t.indices = indices
	t.indptr = indptr
	t.array = arrayFromSlice(data)
	t.o = MakeDataOrder(ColMajor, NonContiguous)
	t.e = StdEng{}

	for _, opt := range opts {
		opt(t)
	}
	return t
}

// CSRFromCoord creates a new Compressed Sparse Row matrix given the coordinates. The data has to be a slice or it panics.
func CSRFromCoord[DT any](shape shapes.Shape, xs, ys []int, data []DT) *CS[DT] {
	t := new(CS)
	t.s = shape
	t.o = NonContiguous
	t.array = arrayFromSlice(data)
	t.e = StdEng{}

	// coord matrix
	cm := &coo[DT]{t.o, xs, ys, t.array}
	sort.Sort(cm)

	r := shape[0]
	c := shape[1]
	if r <= cm.xs[len(cm.xs)-1] || c <= MaxInts(cm.ys...) {
		panic("Cannot create sparse matrix where provided shape is smaller than the implied shape of the data")
	}

	indptr := make([]int, r+1)

	var i, j, tmp int
	for i = 1; i < r+1; i++ {
		for j = tmp; j < len(xs) && xs[j] < i; j++ {

		}
		tmp = j
		indptr[i] = j
	}
	t.indices = ys
	t.indptr = indptr
	return t
}

// CSRFromCoord creates a new Compressed Sparse Column matrix given the coordinates. The data has to be a slice or it panics.
func CSCFromCoord[DT any](shape shapes.Shape, xs, ys []int, data interface{}) *CS[DT] {
	t := new(CS)
	t.s = shape
	t.o = MakeDataOrder(NonContiguous, ColMajor)
	t.array = arrayFromSlice(data)
	t.e = StdEng{}

	// coord matrix
	cm := &coo[DT]{t.o, xs, ys, t.array}
	sort.Sort(cm)

	r := shape[0]
	c := shape[1]

	// check shape
	if r <= MaxInts(cm.xs...) || c <= cm.ys[len(cm.ys)-1] {
		panic("Cannot create sparse matrix where provided shape is smaller than the implied shape of the data")
	}

	indptr := make([]int, c+1)

	var i, j, tmp int
	for i = 1; i < c+1; i++ {
		for j = tmp; j < len(ys) && ys[j] < i; j++ {

		}
		tmp = j
		indptr[i] = j
	}
	t.indices = xs
	t.indptr = indptr
	return t
}
func (t *CS[DT]) IsNil() bool          { return t == nil }
func (t *CS[DT]) Shape() shapes.Shape  { return t.s }
func (t *CS[DT]) Strides() []int       { return nil }
func (t *CS[DT]) Dtype() dtype.Dtype   { return t.t }
func (t *CS[DT]) Dims() int            { return 2 }
func (t *CS[DT]) Size() int            { return t.s.TotalSize() }
func (t *CS[DT]) DataSize() int        { return len(t.data) }
func (t *CS[DT]) Engine() Engine       { return t.e }
func (t *CS[DT]) DataOrder() DataOrder { return t.o }
