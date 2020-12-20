package tensor

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor/internal/storage"
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
		newShape, newRepeats, newAxis, size, err := e.denseRepeatCheck(t, axis, repeats)
		if err != nil {
			return nil, err
		}
		rr := recycledDense(t.Dtype(), newShape, WithEngine(StdEng{}))
		return e.denseRepeat(tt, rr, newShape, newAxis, size, newRepeats)
	default:
		return nil, errors.Errorf("NYI")
	}
}

// RepeatReuse is like Repeat, but with a provided reuse Tensor. The reuseTensor must be of the same type as the input t.
func (e StdEng) RepeatReuse(t Tensor, reuse Tensor, axis int, repeats ...int) (Tensor, error) {
	switch tt := t.(type) {
	case DenseTensor:
		newShape, newRepeats, newAxis, size, err := e.denseRepeatCheck(t, axis, repeats)
		if err != nil {
			return nil, err
		}

		rr, ok := reuse.(DenseTensor)
		if !ok {
			return nil, errors.Errorf("t is a DenseTensor but reuse is of %T", reuse)
		}
		if !reuse.Shape().Eq(newShape) {
			return nil, errors.Errorf("Reuse shape is %v. Expected shape is %v", reuse.Shape(), newShape)
		}
		return e.denseRepeat(tt, rr, newShape, newAxis, size, newRepeats)
	default:
		return nil, errors.Errorf("NYI")
	}
}

func (StdEng) denseRepeatCheck(t Tensor, axis int, repeats []int) (newShape Shape, newRepeats []int, newAxis, size int, err error) {
	if newShape, newRepeats, size, err = t.Shape().Repeat(axis, repeats...); err != nil {
		return nil, nil, -1, -1, errors.Wrap(err, "Unable to get repeated shape")
	}
	newAxis = axis
	if axis == AllAxes {
		newAxis = 0
	}

	return
}

func (StdEng) denseRepeat(t, reuse DenseTensor, newShape Shape, axis, size int, repeats []int) (retVal DenseTensor, err error) {
	d, err := assertDense(reuse)
	if err != nil {
		return nil, errors.Wrapf(err, "Repeat reuse is not a *Dense")
	}
	var outers int
	if t.IsScalar() {
		outers = 1
	} else {
		outers = ProdInts(t.Shape()[0:axis])
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
	// e can never be nil. Error would have occurred elsewhere
	var ok bool
	if fce, ok = e.(fastcopier); ok {
		fastCopy = true
	}

	// In this case, let's not implement the fast copy to keep the code readable
	if ms, ok := t.(MaskedTensor); ok && ms.IsMasked() {
		fastCopy = false
	}

	// if d is not a fastcopier, then we also cannot use fast copy
	if _, ok := d.Engine().(fastcopier); !ok {
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

func (e StdEng) fastCopyDenseRepeat(src DenseTensor, dest *Dense, outers, size, stride, newStride int, repeats []int) error {
	sarr := src.arr()
	darr := dest.arr()

	var destStart, srcStart int
	for i := 0; i < outers; i++ {
		// faster shortcut for common case.
		//
		// Consider a case where:
		// 	a := ⎡ 1 ⎤
		//	     ⎢ 2 ⎥
		//	     ⎢ 3 ⎥
		//	     ⎣ 4 ⎦
		// a has a shape of (4, 1). it is a *Dense.
		//
		// Now assume we want to repeat it on axis 1, 3 times. We want to repeat it into `b`,
		// which is already allocated and zeroed, as shown below
		//
		// 	b := ⎡ 0 0 0 ⎤
		//	     ⎢ 0 0 0 ⎥
		//	     ⎢ 0 0 0 ⎥
		//	     ⎣ 0 0 0 ⎦
		//
		// Now, both `a` and `b` have a stride of 1.
		//
		// The desired result is:
		// 	b := ⎡ 1 1 1 ⎤
		//	     ⎢ 2 2 2 ⎥
		//	     ⎢ 3 3 3 ⎥
		//	     ⎣ 4 4 4 ⎦
		///
		// Observe that this is simply broadcasting (copying) a[0] (a scalar value) to the row b[0], and so on and so forth.
		// This can be done without knowing the full type - we simply copy the bytes over.
		if stride == 1 && newStride == 1 {
			for sz := 0; sz < size; sz++ {
				tmp := repeats[sz]

				// first we get the bounds of the src and the dest
				// the srcStart and destStart are the indices assuming a flat array of []T
				// we need to get the byte slice equivalent.
				bSrcStart := srcStart * int(sarr.t.Size())
				bSrcEnd := (srcStart + stride) * int(sarr.t.Size())
				bDestStart := destStart * int(darr.t.Size())
				bDestEnd := (destStart + tmp) * int(darr.t.Size())

				// then we get the data as a slice of raw bytes
				sBS := sarr.Header.Raw
				dBS := darr.Header.Raw

				// recall that len(src) < len(dest)
				// it's easier to understand if we define the ranges.
				// Less prone to errors.
				sRange := sBS[bSrcStart:bSrcEnd]
				dRange := dBS[bDestStart:bDestEnd]

				// finally we copy things.
				for i := 0; i < len(dRange); i += len(sRange) {
					copy(dRange[i:], sRange)
				}
				srcStart += stride
				destStart += tmp
			}

			// we can straightaway broadcast

			continue
		}

		for j := 0; j < size; j++ {
			var tmp int
			tmp = repeats[j]
			var tSlice array

			tSlice = sarr.slice(srcStart, src.len())

			for k := 0; k < tmp; k++ {
				if srcStart >= src.len() || destStart+stride > dest.len() {
					break
				}

				dSlice := darr.slice(destStart, destStart+newStride)

				// THIS IS AN OPTIMIZATION. REVISIT WHEN NEEDED.
				storage.Copy(dSlice.t.Type, &dSlice.Header, &tSlice.Header)

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

	retVal := recycledDense(a.Dtype(), newShape, WithEngine(e))
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

		// keep dims after slicing
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
			} else if T.Shape()[axis] == 1 {
				if err := v.unsqueeze(axis); err != nil {
					return nil, errors.Wrapf(err, "Unable to keep dims after slicing a shape %v on axis %d where the size is 1", T.Shape(), axis)
				}
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
