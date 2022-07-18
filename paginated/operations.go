package paginated

import (
	"fmt"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"time"

	"gorgonia.org/tensor"
)

// Slice will return a `*View` of the tensor.
func (p *Tensor) Slice(s ...tensor.Slice) (tensor.View, error) {
	if len(s) != p.Dims() {
		return &View{}, ErrDims
	}

	return NewView(p, s...), nil
}

// At will retrieve and return the value in the tensor at the given
// coordinates. Note: coordinates indexing starts with 0.
func (p *Tensor) At(coord ...int) (interface{}, error) {
	page, err := p.page(coord...)
	if err != nil {
		return nil, err
	}

	if p.cache.Contains(page.Id) {
		d, ok := p.cache.Get(page.Id)
		if ok {
			return d.(Array).At(p.pageIndex(coord...))
		}
	}

	err = p.Swap(page.Id)
	if err != nil {
		return nil, err
	}
	d, _ := p.cache.Get(page.Id)

	return d.(Array).At(p.pageIndex(coord...))
}

// SetAt will set the provided value at the provided coordinates
func (p *Tensor) SetAt(v interface{}, coord ...int) error {
	page, err := p.page(coord...)
	if err != nil {
		return err
	}

	if p.cache.Contains(page.Id) {
		d, ok := p.cache.Get(page.Id)
		if ok {
			return d.(Array).SetAt(v, p.pageIndex(coord...))
		}
	}

	err = p.Swap(page.Id)
	if err != nil {
		return err
	}
	d, _ := p.cache.Get(page.Id)

	return d.(Array).SetAt(v, p.pageIndex(coord...))
}

// Reshape will be called when adding data
func (p *Tensor) Reshape(input ...int) error {
	newShape := tensor.Shape(input)
	if newShape.TotalSize() != p.Size() {
		return ErrSize
	}

	p.dims = newShape

	return nil
}

// T will transpose the paginated tensor.
// The axes arguments supplied should specify the new order
// of the dimensions. For example, (1,0,2) will flip the first
// and second dimensions positions.
// If the axes arguments do not equal the number of dimensions in the Tensor
// then an `ErrDims` value will be returned.
// If any of the axes arguments are out of the bounds [0:Dims()-1], then
// an `ErrIndex` value will be returned.
func (p *Tensor) T(axes ...int) error {
	if len(axes) != p.Dims() {
		return ErrDims
	}

	newDims := make(tensor.Shape, p.Dims(), p.Dims())
	for i, v := range axes {
		if v >= p.Dims() {
			return ErrIndex
		}

		newDims[v] = p.dims[i]
	}

	p.old = p.dims
	p.transposed = true
	p.dims = newDims

	return nil
}

// UT ...
func (p *Tensor) UT() {
	if !p.transposed {
		return
	}

	p.dims = p.old
	p.old = tensor.Shape{}
	p.transposed = false
}

// Transpose is not implemented. Please use the `T()` (tranpose)
// and `UT()` (un-transpose) methods instead.
// This will only return an `ErrNotImplemented` value.
func (p *Tensor) Transpose() error {
	return ErrNotImplemented
}

// Apply will create a new paginated tensor stored in a subdirectory at the `applyPath`
// It will be the result of applying the provided function on the values of the
// Tensor.
func (p *Tensor) Apply(fn interface{}, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	fncName := runtime.FuncForPC(reflect.ValueOf(fn).Pointer()).Name()
	name := fmt.Sprintf("%s-%s", fncName, time.Now().Format(stdTimeFmt))
	dir := p.applyPath + name
	err := os.MkdirAll(dir, os.ModePerm)
	if err != nil {
		return nil, err
	}

	applied := &Tensor{
		pages:      make(pages, len(p.pages), len(p.pages)),
		pageSize:   p.pageSize,
		dataOrder:  p.dataOrder,
		dims:       p.dims,
		dtype:      p.dtype,
		transposed: p.transposed,
		engine:     p.engine,
		basePath:   dir + "/",
		fileFormat: p.fileFormat,
	}
	err = applied.init()
	if err != nil {
		return applied, err
	}

	for i := 0; i < len(p.pages); i++ {
		// pages
		oldPage := p.pages[i]
		newPage := &page{
			Id:     oldPage.Id,
			File:   applied.basePath + strconv.Itoa(oldPage.Id) + "." + p.fileExtension(),
			Bounds: oldPage.Bounds,
		}
		applied.pages = append(applied.pages, newPage)

		// apply and copy
		if p.cache.Contains(oldPage.Id) {
			// in-memory
			err := p.flush(oldPage)
			if err != nil {
				return applied, err
			}

			d, _ := p.cache.Get(oldPage.Id)
			err = d.(Array).Apply(fn)
			if err != nil {
				return applied, err
			}

			err = p.flush(newPage)
			if err != nil {
				return applied, err
			}

			err = p.load(oldPage)
			if err != nil {
				return applied, err
			}
		} else {
			// not in-memory
			err := p.Swap(oldPage.Id)
			if err != nil {
				return applied, err
			}

			d, _ := p.cache.Get(oldPage)
			err = d.(Array).Apply(fn)
			if err != nil {
				return applied, err
			}

			err = p.flush(newPage)
			if err != nil {
				return applied, err
			}

			err = p.load(oldPage)
			if err != nil {
				return applied, err
			}
		}

		applied.pages = append(applied.pages, newPage)
	}

	return applied, nil
}
