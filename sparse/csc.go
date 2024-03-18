package sparse

import (
	"fmt"
	"sort"

	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/array"
)

var (
	_ tensor.Basic[float64] = &CS[float64]{}
	//_ tensor.Tensor[float64, *CS[float64]] = &CS[float64]{}
)

// coo is an internal representation of the Coordinate type sparse matrix.
// It's not exported because you probably shouldn't be using it.
// Instead, constructors for the *CS type supports using a coordinate as an input.
type coo[DT any] struct {
	o      DataOrder
	xs, ys []int
	array.Array[DT]
}

func (c *coo[DT]) Len() int { return c.DataSize() }
func (c *coo[DT]) Less(i, j int) bool {
	if c.o.IsColMajor() {
		return c.colMajorLess(i, j)
	}
	return c.rowMajorLess(i, j)
}
func (c *coo[DT]) Swap(i, j int) {
	c.xs[i], c.xs[j] = c.xs[j], c.xs[i]
	c.ys[i], c.ys[j] = c.ys[j], c.ys[i]
	cdata := c.Data()
	cdata[i], cdata[j] = cdata[j], cdata[i]
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

	array.Array[DT]
}

// NewCSR creates a new Compressed Sparse Row matrix. The data has to be a slice or it panics.
func NewCSR[DT any](indices, indptr []int, data []DT, opts ...ConsOpt) *CS[DT] {
	t := new(CS[DT])
	t.indices = indices
	t.indptr = indptr
	t.Array = array.Make(data)
	t.o = tensor.NonContiguous
	t.e = defaultEngine[DT]()

	// for _, opt := range opts {
	// 	opt(t)
	// }
	return t
}

// NewCSC creates a new Compressed Sparse Column matrix. The data has to be a slice, or it panics.
func NewCSC[DT any](indices, indptr []int, data []DT, opts ...ConsOpt) *CS[DT] {
	t := new(CS[DT])
	t.indices = indices
	t.indptr = indptr
	t.Array = array.Make(data)
	t.o = tensor.MakeDataOrder(tensor.ColMajor, tensor.NonContiguous)
	t.e = defaultEngine[DT]()

	// for _, opt := range opts {
	// 	opt(t)
	// }
	return t
}

// CSRFromCoord creates a new Compressed Sparse Row matrix given the coordinates. The data has to be a slice or it panics.
func CSRFromCoord[DT any](shape shapes.Shape, xs, ys []int, data []DT) *CS[DT] {
	t := new(CS[DT])
	t.s = shape
	t.o = tensor.NonContiguous
	t.Array = array.Make(data)
	t.e = defaultEngine[DT]()

	// coord matrix
	cm := &coo[DT]{
		o:     t.o,
		xs:    xs,
		ys:    ys,
		Array: t.Array, // TODO: check. should this be a new array?
	}

	sort.Sort(cm)

	r := shape[0]
	c := shape[1]
	if r <= cm.xs[len(cm.xs)-1] || c <= internal.MaxS(cm.ys...) {
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
func CSCFromCoord[DT any](shape shapes.Shape, xs, ys []int, data []DT) *CS[DT] {
	t := new(CS[DT])
	t.s = shape
	t.o = tensor.MakeDataOrder(tensor.NonContiguous, tensor.ColMajor)
	t.Array = array.Make(data)
	t.e = defaultEngine[DT]()

	// coord matrix
	cm := &coo[DT]{
		o:     t.o,
		xs:    xs,
		ys:    ys,
		Array: t.Array, // TODO: check. Should this be a new  array? A sort happens here.
	}
	sort.Sort(cm)

	r := shape[0]
	c := shape[1]

	// check shape
	if r <= internal.MaxS(cm.xs...) || c <= cm.ys[len(cm.ys)-1] {
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
func (t *CS[DT]) IsNil() bool                { return t == nil }
func (t *CS[DT]) Shape() shapes.Shape        { return t.s }
func (t *CS[DT]) Strides() []int             { return nil }
func (t *CS[DT]) Dtype() dtype.Dtype         { return t.t }
func (t *CS[DT]) Dims() int                  { return 2 }
func (t *CS[DT]) Size() int                  { return t.s.TotalSize() }
func (t *CS[DT]) Engine() Engine             { return t.e }
func (t *CS[DT]) DataOrder() DataOrder       { return t.o }
func (t *CS[DT]) Flags() MemoryFlag          { return t.f }
func (t *CS[DT]) IsManuallyManaged() bool    { return t.f.IsManuallyManaged() }
func (t *CS[DT]) IsNativelyAccessible() bool { return t.f.IsNativelyAccessible() }
func (t *CS[DT]) IsMaterializable() bool     { return false }
func (t *CS[DT]) Memset(v DT) error {
	if t.f.IsNativelyAccessible() {
		t.Array.Memset(v)
		return nil
	}
	return t.e.Memset(t, v)
}

func (t *CS[DT]) SetAt(v DT, coords ...int) error {
	panic("NYI")
}

func (t *CS[DT]) Info() *tensor.AP { return nil } // tmp. Try to reuse the AP data structure!

func (t *CS[DT]) Iterator() tensor.Iterator { panic("NYI") } // TODO
func (t *CS[DT]) RequiresIterator() bool    { return true }

func (t *CS[DT]) AlikeAsType(dt dtype.Dtype, opts ...ConsOpt) tensor.DescWithStorage {
	if !t.Dtype().Eq(dt) {
		panic(fmt.Sprintf("Not yet implemented - creating a *CS of %v", dt))
	}
	return &CS[DT]{}
}

func (t *CS[DT]) AlikeAsDescWithStorage(opts ...ConsOpt) tensor.DescWithStorage {
	panic("NYI")
}

func (t *CS[DT]) Clone() *CS[DT]                 { panic("NYI") }
func (t *CS[DT]) CloneAsBasic() tensor.Basic[DT] { panic("NYI") }
func (t *CS[DT]) Reshape(shp ...int) error {
	panic("NYI")
}
func (t *CS[DT]) Eq(other *CS[DT]) bool { panic("NYI") }

func (t *CS[DT]) Restore() { panic("NYI") }

func (t *CS[DT]) SetDataOrder(o DataOrder) { panic("NYI") }

func (t *CS[DT]) Unsqueeze(axis int) error { panic("NYI") }

func (t *CS[DT]) Zero() {
	if !t.IsNativelyAccessible() {
		t.e.Memclr(t)
		return
	}
	t.Array.Memclr()
}

func (t *CS[DT]) Alike(opts ...ConsOpt) *CS[DT] {
	panic("NYI")
}

func (t *CS[DT]) AlikeAsBasic(opts ...ConsOpt) tensor.Basic[DT] { panic("NYI") }

func (t *CS[DT]) Apply(fn any, opts ...FuncOpt) (*CS[DT], error) { panic("NYI") }

func (t *CS[DT]) Reduce(fn any, defaultValue DT, opts ...FuncOpt) (*CS[DT], error) { panic("NYI") }

func (t *CS[DT]) Scan(fn func(a, b DT) DT, axis int, opts ...FuncOpt) (*CS[DT], error) { panic("NYI") }

func (t *CS[DT]) Dot(reductionFn, elwiseFn func(DT, DT) DT, other *CS[DT], opts ...FuncOpt) (*CS[DT], error) {
	panic("NYI")
}
func (t *CS[DT]) Repeat(axis int, repeats ...int) (*CS[DT], error) { panic("NYI") }

func (t *CS[DT]) Materialize() (*CS[DT], error) { panic("NYI") }
