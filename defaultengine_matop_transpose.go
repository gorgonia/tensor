// +build !inplacetranspose

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

	orig := a.(*Dense).Mask()
	tmp := make([]bool, len(orig))

	it := newFlatIterator(a.Info())
	var j int
	for i, err := it.Next(); err == nil; i, err = it.Next() {
		tmp[j] = orig[i]
		j++
	}
	copy(orig, tmp)
}

func (e StdEng) denseTranspose1(a DenseTensor, expStrides []int) {
	var tmpArr array
	e.makeArray(&tmpArr, a.Dtype(), a.Size())
	u8s := tmpArr.Uint8s()

	orig := a.hdr().Uint8s()
	it := newFlatIterator(a.Info())
	var j int
	for i, err := it.Next(); err == nil; i, err = it.Next() {
		u8s[j] = orig[i]
		j++
	}
	copy(orig, u8s)
}

func (e StdEng) denseTranspose2(a DenseTensor, expStrides []int) {
	var tmpArr array
	e.makeArray(&tmpArr, a.Dtype(), a.Size())
	u16s := tmpArr.Uint16s()

	orig := a.hdr().Uint16s()
	it := newFlatIterator(a.Info())
	var j int
	for i, err := it.Next(); err == nil; i, err = it.Next() {
		u16s[j] = orig[i]
		j++
	}
	copy(orig, u16s)
}

func (e StdEng) denseTranspose4(a DenseTensor, expStrides []int) {
	var tmpArr array
	e.makeArray(&tmpArr, a.Dtype(), a.Size())
	u32s := tmpArr.Uint32s()

	orig := a.hdr().Uint32s()
	it := newFlatIterator(a.Info())
	var j int
	for i, err := it.Next(); err == nil; i, err = it.Next() {
		u32s[j] = orig[i]
		j++
	}
	copy(orig, u32s)
}

func (e StdEng) denseTranspose8(a DenseTensor, expStrides []int) {
	var tmpArr array
	e.makeArray(&tmpArr, a.Dtype(), a.Size())
	u64s := tmpArr.Uint64s()

	orig := a.hdr().Uint64s()
	it := newFlatIterator(a.Info())
	var j int
	for i, err := it.Next(); err == nil; i, err = it.Next() {
		u64s[j] = orig[i]
		j++
	}
	copy(orig, u64s)
}

func (e StdEng) denseTransposeString(a DenseTensor, expStrides []int) {
	var tmpArr array
	e.makeArray(&tmpArr, a.Dtype(), a.Size())
	strs := tmpArr.Strings()

	orig := a.hdr().Strings()
	it := newFlatIterator(a.Info())
	var j int
	for i, err := it.Next(); err == nil; i, err = it.Next() {
		strs[j] = orig[i]
		j++
	}
	copy(orig, strs)
}

func (e StdEng) denseTransposeArbitrary(a DenseTensor, expStrides []int) {
	rtype := a.rtype()
	typeSize := int(rtype.Size())
	var tmpArr array
	e.makeArray(&tmpArr, a.Dtype(), a.Size())
	// arbs := storage.AsByteSlice(tmpArr.hdr(), rtype)
	arbs := tmpArr.byteSlice()

	orig := a.hdr().Raw
	it := newFlatIterator(a.Info())
	var j int
	for i, err := it.Next(); err == nil; i, err = it.Next() {
		srcStart := i * typeSize
		srcEnd := srcStart + typeSize
		dstStart := j * typeSize
		dstEnd := dstStart + typeSize

		copy(arbs[dstStart:dstEnd], orig[srcStart:srcEnd])
		j++
	}
	copy(orig, arbs)
}
