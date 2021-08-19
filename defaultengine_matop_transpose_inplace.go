// +build inplacetranspose

package tensor

import (
	"context"

	"github.com/pkg/errors"
)

func (e StdEng) Transpose(ctx context.Context, a Tensor, expStrides []int) error {
	if err := handleCtx(ctx); err != nil {
		return err
	}

	if !a.IsNativelyAccessible() {
		return errors.Errorf("Cannot Transpose() on non-natively accessible tensor")
	}
	if dt, ok := a.(DenseTensor); ok {
		e.denseTranspose(dt, expStrides)
		return nil
	}
	return errors.Errorf("Tranpose for tensor of %T not supported", a)
}

func (e StdEng) denseTranspose(a DenseTensor, expStrides []int) {
	if a.rtype() == String.Type {
		e.denseTransposeString(a, expStrides)
		return
	}

	e.transposeMask(a)

	switch a.rtype().Size() {
	case 1:
		e.denseTranspose1(a, expStrides)
	case 2:
		e.denseTranspose2(a, expStrides)
	case 4:
		e.denseTranspose4(a, expStrides)
	case 8:
		e.denseTranspose8(a, expStrides)
	default:
		e.denseTransposeArbitrary(a, expStrides)
	}
}

func (e StdEng) transposeMask(a DenseTensor) {
	if !a.(*Dense).IsMasked() {
		return
	}

	shape := a.Shape()
	if len(shape) != 2 {
		// TODO(poopoothegorilla): currently only two dimensions are implemented
		return
	}
	n, m := shape[0], shape[1]
	mask := a.(*Dense).Mask()
	size := len(mask)

	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1)

	for i := 0; i < size; i++ {
		srci := i
		if track.IsSet(srci) {
			continue
		}
		srcv := mask[srci]
		for {
			oc := srci % n
			or := (srci - oc) / n
			desti := oc*m + or

			if track.IsSet(desti) {
				break
			}
			track.Set(desti)
			destv := mask[desti]
			mask[desti] = srcv
			srci = desti
			srcv = destv
		}
	}
}

func (e StdEng) denseTranspose1(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	var saved, tmp byte
	var i int

	data := a.hdr().Uint8s()
	if len(data) < 4 {
		return
	}
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			data[i] = saved
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		tmp = data[i]
		data[i] = saved
		saved = tmp

		i = dest
	}
}

func (e StdEng) denseTranspose2(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	var saved, tmp uint16
	var i int

	data := a.hdr().Uint16s()
	if len(data) < 4 {
		return
	}
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			data[i] = saved
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		tmp = data[i]
		data[i] = saved
		saved = tmp

		i = dest
	}
}

func (e StdEng) denseTranspose4(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	var saved, tmp uint32
	var i int

	data := a.hdr().Uint32s()
	if len(data) < 4 {
		return
	}
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			data[i] = saved
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		tmp = data[i]
		data[i] = saved
		saved = tmp

		i = dest
	}
}

func (e StdEng) denseTranspose8(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	var saved, tmp uint64
	var i int

	data := a.hdr().Uint64s()
	if len(data) < 4 {
		return
	}
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)
		if track.IsSet(i) && track.IsSet(dest) {
			data[i] = saved
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		// log.Printf("i: %d start %d, end %d | tmp %v saved %v", i, start, end, tmp, saved)
		tmp = data[i]
		data[i] = saved
		saved = tmp

		i = dest
	}
}

func (e StdEng) denseTransposeString(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	var saved, tmp string
	var i int

	data := a.hdr().Strings()
	if len(data) < 4 {
		return
	}
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			data[i] = saved
			saved = ""
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		tmp = data[i]
		data[i] = saved
		saved = tmp

		i = dest
	}
}

func (e StdEng) denseTransposeArbitrary(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()
	rtype := a.rtype()
	typeSize := int(rtype.Size())

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	saved := make([]byte, typeSize, typeSize)
	tmp := make([]byte, typeSize, typeSize)
	var i int
	data := a.arr().Raw
	if len(data) < 4*typeSize {
		return
	}
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)
		start := typeSize * i
		end := start + typeSize

		if track.IsSet(i) && track.IsSet(dest) {
			copy(data[start:end], saved)
			for i := range saved {
				saved[i] = 0
			}
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		copy(tmp, data[start:end])
		copy(data[start:end], saved)
		copy(saved, tmp)
		i = dest
	}
}
