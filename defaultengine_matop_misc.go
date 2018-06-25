package tensor

import "github.com/pkg/errors"

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

		if v.IsVector() && T.IsMatrix() && axis == 0 {
			v.reshape(v.shape[0], 1)
		}

		if err = assignArray(v, T); err != nil {
			return nil, errors.Wrap(err, "Unable to assignArray in denseConcat")
		}
		start = end
	}

	return retVal, nil
}

func (e StdEng) Diagonal(a DenseTensor) (retVal Tensor, err error) {
	if a.Dims() != 2 {
		err = errors.Errorf(dimMismatch, 2, a.Dims())
		return
	}

	if err = typeclassCheck(a.Dtype(), numberTypes); err != nil {
		return nil, errors.Wrap(err, "Trace")
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
		return nil, errors.Errorf(typeNYI, "Arbitrary sized diag")
	}
	return b, nil
}
