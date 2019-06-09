package tensor

import (
	"github.com/pkg/errors"
)

var (
	_ Diager = StdEng{}
)

type fastcopier interface {
	fastCopyDenseRepeat(t DenseTensor, d *Dense, outers, size, stride, newStride int, repeats []int) error
}

// Repeat ...
func (e StdEng) Repeat(t Tensor, axis int, repeats ...int) (Tensor, error) {
	switch tt := t.(type) {
	case DenseTensor:
		return e.denseRepeat(tt, axis, repeats)
	default:
		return nil, errors.Errorf("NYI")
	}
}

func (StdEng) denseRepeat(t DenseTensor, axis int, repeats []int) (retVal DenseTensor, err error) {
	var newShape Shape
	var size int
	if newShape, repeats, size, err = t.Shape().Repeat(axis, repeats...); err != nil {
		return nil, errors.Wrap(err, "Unable to get repeated shape")
	}

	if axis == AllAxes {
		axis = 0
	}

	d := recycledDense(t.Dtype(), newShape)

	var outers int
	if t.IsScalar() {
		outers = 1
	} else {
		outers = ProdInts(t.Shape()[0:axis])
		if outers == 0 {
			outers = 1
		}
	}

	var stride, newStride int
	if newShape.IsVector() || t.IsVector() {
		stride = 1 // special case because CalcStrides() will return []int{1} as the strides for a vector
	} else {
		stride = t.ostrides()[axis]
	}

	if newShape.IsVector() {
		newStride = 1
	} else {
		newStride = d.ostrides()[axis]
	}

	var destStart, srcStart int
	// fastCopy is not bypassing the copyDenseSliced method to populate the output tensor
	var fastCopy bool
	var fce fastcopier
	// we need an engine for fastCopying...
	e := t.Engine()
	// e can never be nil. Error would have occured elsewhere
	var ok bool
	if fce, ok = e.(fastcopier); ok {
		fastCopy = true
	}

	// In this case, let's not implement the fast copy to keep the code readable
	if ms, ok := t.(MaskedTensor); ok && ms.IsMasked() {
		fastCopy = false
	}

	if fastCopy {
		if err := fce.fastCopyDenseRepeat(t, d, outers, size, stride, newStride, repeats); err != nil {
			return nil, err
		}
		return d, nil
	}

	for i := 0; i < outers; i++ {
		for j := 0; j < size; j++ {
			var tmp int
			tmp = repeats[j]

			for k := 0; k < tmp; k++ {
				if srcStart >= t.len() || destStart+stride > d.len() {
					break
				}
				copyDenseSliced(d, destStart, d.len(), t, srcStart, t.len())
				destStart += newStride
			}
			srcStart += stride
		}
	}
	return d, nil
}

func (StdEng) fastCopyDenseRepeat(t DenseTensor, d *Dense, outers, size, stride, newStride int, repeats []int) error {
	var destStart, srcStart int
	for i := 0; i < outers; i++ {
		for j := 0; j < size; j++ {
			var tmp int
			tmp = repeats[j]
			var tSlice array
			tSlice = t.arr().slice(srcStart, t.len())

			for k := 0; k < tmp; k++ {
				if srcStart >= t.len() || destStart+stride > d.len() {
					break
				}
				dSlice := d.arr().slice(destStart, d.len())
				if err := t.Engine().Memcpy(&dSlice, &tSlice); err != nil {
					return err
				}
				destStart += newStride
			}
			srcStart += stride
		}
	}
	return nil
}

// Concat tensors
func (e StdEng) Concat(t Tensor, axis int, others ...Tensor) (retVal Tensor, err error) {
	switch tt := t.(type) {
	case DenseTensor:
		var denses []DenseTensor
		if denses, err = tensorsToDenseTensors(others); err != nil {
			return nil, errors.Wrap(err, "Concat failed")
		}
		return e.denseConcat(tt, axis, denses)
	default:
		return nil, errors.Errorf("NYI")
	}
}

func (e StdEng) denseConcat(a DenseTensor, axis int, Ts []DenseTensor) (DenseTensor, error) {
	ss := make([]Shape, len(Ts))
	var err error
	var isMasked bool
	for i, T := range Ts {
		ss[i] = T.Shape()
		if mt, ok := T.(MaskedTensor); ok {
			isMasked = isMasked || mt.IsMasked()
		}
	}

	var newShape Shape
	if newShape, err = a.Shape().Concat(axis, ss...); err != nil {
		return nil, errors.Wrap(err, "Unable to find new shape that results from concatenation")
	}

	retVal := recycledDense(a.Dtype(), newShape)
	if isMasked {
		retVal.makeMask()
	}

	all := make([]DenseTensor, len(Ts)+1)
	all[0] = a
	copy(all[1:], Ts)

	// TODO: OPIMIZATION
	// When (axis == 0 && a is row major and all others is row major) || (axis == last axis of A && all tensors are colmajor)
	// just flat copy
	//

	// isOuter  is true when the axis is the outermost axis
	// isInner is true when the axis is the inner most axis
	isOuter := axis == 0
	isInner := axis == (a.Shape().Dims() - 1)

	// special case
	var start, end int
	for _, T := range all {
		end += T.Shape()[axis]
		slices := make([]Slice, axis+1)
		slices[axis] = makeRS(start, end)

		var v *Dense
		if v, err = sliceDense(retVal, slices...); err != nil {
			return nil, errors.Wrap(err, "Unable to slice DenseTensor while performing denseConcat")
		}

		switch {
		case v.IsVector() && T.IsMatrix() && axis == 0:
			v.reshape(v.shape[0], 1)
		case T.IsRowVec() && axis == 0:
			T.reshape(T.Shape()[1])
		case v.Shape().IsScalarEquiv() && T.Shape().IsScalarEquiv():
			copyArray(v.arrPtr(), T.arrPtr())
			if mt, ok := T.(MaskedTensor); ok {
				copy(v.mask, mt.Mask())
			}
			continue
		default:
			diff := retVal.Shape().Dims() - v.Shape().Dims()
			if diff > 0 && isOuter {
				newShape := make(Shape, v.Shape().Dims()+diff)
				for i := 0; i < diff; i++ {
					newShape[i] = 1
				}
				copy(newShape[diff:], v.Shape())
				v.reshape(newShape...)
			} else if diff > 0 && isInner {
				newShape := v.Shape().Clone()
				newStrides := v.strides
				for i := 0; i < diff; i++ {
					newShape = append(newShape, 1)
					newStrides = append(newStrides, 1)
				}
				v.shape = newShape
				v.strides = newStrides
			}
		}

		var vmask, Tmask []bool
		vmask = v.mask
		v.mask = nil
		if mt, ok := T.(MaskedTensor); ok && mt.IsMasked() {
			Tmask = mt.Mask()
			mt.SetMask(nil)

		}

		if err = assignArray(v, T); err != nil {
			return nil, errors.Wrap(err, "Unable to assignArray in denseConcat")
		}
		// if it's a masked tensor, we copy the mask as well
		if Tmask != nil {
			if vmask != nil {
				if cap(vmask) < len(Tmask) {
					vmask2 := make([]bool, len(Tmask))
					copy(vmask2, vmask)
					vmask = vmask2
				}
				copy(vmask, Tmask)
				v.SetMask(vmask)
			}
			// mt.SetMask(Tmask)
		}

		start = end
	}

	return retVal, nil
}

// Diag ...
func (e StdEng) Diag(t Tensor) (retVal Tensor, err error) {
	a, ok := t.(DenseTensor)
	if !ok {
		return nil, errors.Errorf("StdEng only works with DenseTensor for Diagonal()")
	}

	if a.Dims() != 2 {
		err = errors.Errorf(dimMismatch, 2, a.Dims())
		return
	}

	if err = typeclassCheck(a.Dtype(), numberTypes); err != nil {
		return nil, errors.Wrap(err, "Diagonal")
	}

	rstride := a.Strides()[0]
	cstride := a.Strides()[1]

	r := a.Shape()[0]
	c := a.Shape()[1]

	m := MinInt(r, c)
	stride := rstride + cstride

	b := a.Clone().(DenseTensor)
	b.Zero()

	switch a.rtype().Size() {
	case 1:
		bdata := b.hdr().Uint8s()
		adata := a.hdr().Uint8s()
		for i := 0; i < m; i++ {
			bdata[i] = adata[i*stride]
		}
	case 2:
		bdata := b.hdr().Uint16s()
		adata := a.hdr().Uint16s()
		for i := 0; i < m; i++ {
			bdata[i] = adata[i*stride]
		}
	case 4:
		bdata := b.hdr().Uint32s()
		adata := a.hdr().Uint32s()
		for i := 0; i < m; i++ {
			bdata[i] = adata[i*stride]
		}
	case 8:
		bdata := b.hdr().Uint64s()
		adata := a.hdr().Uint64s()
		for i := 0; i < m; i++ {
			bdata[i] = adata[i*stride]
		}
	default:
		return nil, errors.Errorf(typeNYI, "Arbitrary sized diag", t)
	}
	return b, nil
}
