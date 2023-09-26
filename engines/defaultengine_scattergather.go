package stdeng

import (
	"context"
	"sync"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/axialiter"
	"gorgonia.org/tensor/internal/errors"
)

func (e StdEng[DT, T]) Scatter(ctx context.Context, t T, indices tensor.Basic[int], retVal T) error {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}

	oldShape := t.Shape().Clone()
	oldIndicesShape := t.Shape().Clone()
	retValOldShape := retVal.Shape().Clone()

	defer func() {
		t.Reshape(oldShape...)
		indices.Reshape(oldIndicesShape...)
		retVal.Reshape(retValOldShape...)
	}()

	// reshape everything into a matrix
	t.Reshape(asMat(t.Shape(), t.Dims()-1, true)...)
	indices.Reshape(asMat(indices.Shape(), indices.Dims()-1, true)...)
	retVal.Reshape(asMat(retVal.Shape(), retVal.Dims()-1, true)...)

	// check that indices shape[0] <= t.Shape()[0]
	if indices.Shape()[0] > t.Shape()[0] {
		return errors.Errorf("Scatter: indices shape[0] (%d) > t.Shape()[0] (%d)", indices.Shape()[0], t.Shape()[0])
	}

	// now they are matrices, we can iterate thru them
	var ps []iteratorPair[DT]
	for i := 0; i < indices.Shape()[0]; i++ {
		ait := axialiter.New(t.Info(), 0, i, true)
		iit := axialiter.New(indices.Info(), 0, i, true)

		ps = append(ps, iteratorPair[DT]{ait, iit, i})
	}

	errChan := make(chan error, len(ps))
	var wg sync.WaitGroup
	for i := range ps {
		wg.Add(1)
		// note: be careful not to use `for i, p := range ps`
		// and then use `go p.coiter`.
		// This is because `p` is would not be captured by `go`,
		// thus every `p` would be `ps[len(ps)-1]`.
		go ps[i].coiter(t, indices, retVal, errChan, &wg)
	}
	wg.Wait()
	close(errChan)
	err := <-errChan // maybe get ALL the errors from errChan?
	return err
}

type iteratorPair[DT any] struct {
	a    *axialiter.AxialIterator
	idx  *axialiter.AxialIterator
	axis int
}

func (it *iteratorPair[DT]) coiter(a tensor.Basic[DT], indices tensor.Basic[int], reuse tensor.Basic[DT], errChan chan error, wg *sync.WaitGroup) {
	defer wg.Done()
	ii, err := it.idx.Start()
	if err != nil {
		if err = internal.HandleNoOp(err); err != nil {
			errChan <- err
		}
		return
	}

	iData := indices.Data()
	retStride := reuse.Strides()[0]
	aData := a.Data()
	rData := reuse.Data()

	var ai int
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

reterr:
	if err = internal.HandleNoOp(err); err != nil {
		errChan <- err
		return
	}

}
