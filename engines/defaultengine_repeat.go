package stdeng

import (
	"context"

	"github.com/chewxy/inigo/values/tensor/internal"
)

func (e StdEng[DT, T]) Repeat(ctx context.Context, a, retVal T, axis, size int, repeats []int) error {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}

	return e.denseRepeat(a, retVal, axis, size, repeats)
}

func (_ StdEng[DT, T]) denseRepeat(t, reuse T, axis, size int, repeats []int) (err error) {
	tShape := t.Shape()
	dShape := reuse.Shape()
	var outers int
	if tShape.IsScalar() {
		outers = 1
	} else {
		outers = internal.Prod[int](tShape[0:axis])
	}

	var stride, newStride int
	switch {
	case dShape.IsVector():
		stride = 1
		newStride = 1
	case tShape.IsVector():
		stride = 1 // special case because CalcStrides() will return []int{1} as the strides for a vector
		newStride = reuse.Strides()[axis]
	case tShape.IsScalar():
		stride = 1
		newStride = 1
	default:
		stride = t.Strides()[axis]
		newStride = reuse.Strides()[axis]
	}

	var destStart, srcStart int
	// // fastCopy is not bypassing the copyDenseSliced method to populate the output tensor
	// var fastCopy bool
	// var fce fastcopier
	// // we need an engine for fastCopying...
	// e := t.Engine()
	// // e can never be nil. Error would have occurred elsewhere
	// var ok bool
	// if fce, ok = e.(fastcopier); ok {
	// 	fastCopy = true
	// }

	// if t.RequiresIterator() || reuse.RequiresIterator() {
	// 	fastCopy = false
	// }

	// // In this case, let's not implement the fast copy to keep the code readable
	// // if ms, ok := t.(MaskedTensor); ok && ms.IsMasked() {
	// // 	fastCopy = false
	// // }

	// // if d is not a fastcopier, then we also cannot use fast copy
	// if _, ok := reuse.Engine().(fastcopier); !ok {
	// 	fastCopy = false
	// }

	// if fastCopy {
	// 	if err := fce.fastCopyDenseRepeat(t, reuse, outers, size, stride, newStride, repeats); err != nil {
	// 		return nil, err
	// 	}
	// 	return nil
	// }

	lenT := len(t.Data())
	lenD := len(reuse.Data())
	for i := 0; i < outers; i++ {
		for j := 0; j < size; j++ {
			var tmp int
			tmp = repeats[j]

			for k := 0; k < tmp; k++ {
				if srcStart >= lenT || destStart+stride > lenD {
					break
				}
				copy(reuse.Data()[destStart:lenD], t.Data()[srcStart:lenT])
				destStart += newStride
			}
			srcStart += stride
		}
	}
	return nil
}
