package dense

import (
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

// ToMat64 converts a *Dense[DT] to a gonum.Mat. It allows `UseUnsafe` as an option if `DT` is `float64`. For all other `DT`s the options are ignored.
func ToMat64[DT OrderedNum, T tensor.Basic[DT]](t T, opts ...FuncOpt) (retVal *mat.Dense, err error) {
	// checks:
	if !t.IsNativelyAccessible() {
		return nil, errors.Errorf("Cannot convert *Dense to *mat.Dense. Data is inaccessible")
	}

	if !tensor.IsMatrix(t) {
		// error
		return nil, errors.Errorf("Cannot convert *Dense to *mat.Dense. Expected number of dimensions: <=2, T has got %d dimensions (Shape: %v)", t.Dims(), t.Shape())
	}

	fo := ParseFuncOpts(opts...)
	toCopy := fo.Safe()

	// fix dims
	r := t.Shape()[0]
	c := t.Shape()[1]

	var data []float64
	var v [0]DT
	switch any(v).(type) {
	case [0]float64:
		switch {
		case toCopy && !t.IsMaterializable():
			td := any(t.Data()).([]float64)
			data = make([]float64, len(td))
			copy(data, td)
		case !toCopy && !t.IsMaterializable():
			data = any(t.Data()).([]float64)
		default:
			// use iterators
		}
	default:
		if !t.IsMaterializable() {
			data = convert[float64, DT](t.Data())
		} else {
			// use iterator
			tdata := t.Data()

			it := t.Iterator()
			var next int
			for next, err = it.Next(); err == nil; next, err = it.Next() {
				if err = internal.HandleNoOp(err); err != nil {
					return
				}
				data = append(data, float64(tdata[next]))
			}
			err = nil
		}
	}
	retVal = mat.NewDense(r, c, data)
	return
}

// FromMat64 converts a value of gonum's mat.Dense into a *Dense[DT]. It accepts `tensor.UseUnsafe` as a func opt if `DT` is a float64. If `DT` isn't `float64`, the unsafe option is ignored and a copied version of the backing array is used.
func FromMat64[DT OrderedNum](m *mat.Dense, opts ...FuncOpt) *Dense[DT] {
	r, c := m.Dims()
	fo := ParseFuncOpts(opts...)
	toCopy := fo.Safe()

	var v [0]DT
	switch any(v).(type) {
	case [0]float64:
		var backing []float64
		if toCopy {
			backing = make([]float64, len(m.RawMatrix().Data))
			copy(backing, m.RawMatrix().Data)
		} else {
			backing = m.RawMatrix().Data
		}
		retVal := New[DT](WithBacking(backing), WithShape(r, c))
		return retVal
	default:
		backing := convert[DT, float64](m.RawMatrix().Data)
		retVal := New[DT](WithBacking(backing), WithShape(r, c))
		return retVal
	}

}
