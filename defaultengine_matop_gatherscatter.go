package tensor

import (
	"sync"

	"github.com/pkg/errors"
)

func (e StdEng) Scatter(a, indices Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	fo := ParseFuncOpts(opts...)
	reuse := fo.Reuse()

	maxT, err := Max(indices)
	if err != nil {
		return nil, errors.Wrapf(err, "Cannot find the max of the indices")
	}
	max, ok := maxT.Data().(int)
	if !ok {
		return nil, errors.Errorf("Indices must be of ints. Got %v of %T instead", maxT.Data(), maxT.Data())
	}

	// expected shape
	shp := indices.Shape().Clone()
	shp[len(shp)-1] = max + 1

	switch {
	case reuse == nil && fo.Safe():
		// create reuse
		reuse = New(WithShape(shp...), Of(a.Dtype()))
	case reuse == nil && !fo.Safe():
		// check shape of `a`
	case reuse != nil:
		// check shape of `reuse`
	}

	oldShape := a.Shape().Clone()
	oldIndicesShape := a.Shape().Clone()
	reuseOldShape := reuse.Shape().Clone()
	defer func() { a.Reshape(oldShape...); indices.Reshape(oldIndicesShape...); reuse.Reshape(reuseOldShape...) }()

	switch {
	case indices.Shape().IsVectorLike():
		idx := indices.Data().([]int)
		_ = idx
		// TODO
	default:
		// THIS IS ROW MAJOR ONLY
		// THIS IS DENSE TENSOR ONLY

		a := a.(DenseTensor)
		indices := indices.(DenseTensor)
		reuse := reuse.(DenseTensor)

		// reshape everything into a matrix
		a.Reshape(asMat(a.Shape(), a.Dims()-1, true)...)
		indices.Reshape(asMat(indices.Shape(), indices.Dims()-1, true)...)
		reuse.Reshape(asMat(reuse.Shape(), reuse.Dims()-1, true)...)

		// check that indices' shape[0] is <= a.Shape[0]
		if indices.Shape()[0] > a.Shape()[0] {
			// something is wrong
			return nil, errors.Errorf("Cannot scatter")
		}

		// now they are all matrices, we can iterate thru them
		var ps []iteratorPair
		for i := 0; i < indices.Shape()[0]; i++ {
			ait := AxialIteratorFromDense(a, 0, i, true)
			iit := AxialIteratorFromDense(indices, 0, i, true)

			ps = append(ps, iteratorPair{ait, iit, i})
		}

		errChan := make(chan error, len(ps))
		var wg sync.WaitGroup
		for i := range ps {
			wg.Add(1)
			// note: be careful not to use `for i, p := range ps`
			// and then use `go p.coiter`.
			// This is because `p` is would not be captured by `go`,
			// thus every `p` would be `ps[len(ps)-1]`.
			go ps[i].coiter(a, indices, reuse, errChan, &wg)
		}
		wg.Wait()
		close(errChan)
		err = <-errChan // maybe get ALL the errors from errChan?
		return reuse, err

	}

	panic("NYI")
}

type iteratorPair struct {
	a    *AxialIterator
	idx  *AxialIterator
	axis int
}

func (it *iteratorPair) coiter(a, indices, reuse DenseTensor, errChan chan error, wg *sync.WaitGroup) {
	defer wg.Done()
	ii, err := it.idx.Start()
	if err != nil {
		if err = handleNoOp(err); err != nil {
			errChan <- err
		}
		return
	}

	iData := indices.Data().([]int)
	retStride := reuse.Strides()[0]
	switch {
	case a.Dtype() == Float64 && reuse.Dtype() == Float64:
		aData := a.Data().([]float64)
		rData := reuse.Data().([]float64)

		var ai, ii int
		if ai, err = it.a.Start(); err != nil {
			goto reterr
		}
		if ii, err = it.idx.Start(); err != nil {
			goto reterr
		}
		for {

			idx := iData[ii]
			v := aData[ai]

			rData[it.axis*retStride+idx] = v

			if it.a.Done() || it.idx.Done() {
				break
			}
			if ai, err = it.a.Next(); err != nil {
				break
			}
			if ii, err = it.idx.Next(); err != nil {
				break
			}
		}
	case a.Dtype() == Float32 && reuse.Dtype() == Float32:
		aData := a.Data().([]float32)
		rData := reuse.Data().([]float32)

		var ai, ii int
		if ai, err = it.a.Start(); err != nil {
			goto reterr
		}
		if ii, err = it.idx.Start(); err != nil {
			goto reterr
		}
		for {

			idx := iData[ii]
			v := aData[ai]

			rData[it.axis*retStride+idx] = v

			if it.a.Done() || it.idx.Done() {
				break
			}
			if ai, err = it.a.Next(); err != nil {
				break
			}
			if ii, err = it.idx.Next(); err != nil {
				break
			}
		}

	default:

		// generic
		for ai, err := it.a.Start(); err == nil; ai, err = it.a.Next() {
			if it.idx.Done() {
				break
			}
			idx := iData[ii]
			v := a.arrPtr().Get(ai)
			reuse.Set(it.axis*retStride+idx, v)

			if ii, err = it.idx.Next(); err != nil {
				break
			}
		}
	}

reterr:
	if err = handleNoOp(err); err != nil {
		errChan <- err
		return
	}

}
