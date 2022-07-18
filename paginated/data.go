package paginated

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Zero will set all values in the tensor to their zero value.
// This should be used very cautiously with paginated tensors
// as all persisted data will be zeroed as well.
// This function will panic if any error occurs.
func (p *Tensor) Zero() {
	for i := 0; i < len(p.pages); i++ {
		page, err := p.pages.get(i)
		if err != nil {
			panic(err)
		}

		if p.cache.Contains(page.Id) {
			d, _ := p.cache.Get(page.Id)
			d.(Array).Zero()
		} else {
			err := p.Swap(page.Id)
			if err != nil {
				panic(err)
			}

			d, _ := p.cache.Get(page.Id)
			d.(Array).Zero()
		}

		err = p.flush(page)
		if err != nil {
			panic(err)
		}
	}
}

// Memset will set all values in the tensor to the provided
// value.
// This should be used very cautiously with paginated tensors
// as all persisted data will be set to provided value as well.
func (p *Tensor) Memset(input interface{}) error {
	for i := 0; i < len(p.pages); i++ {
		page, err := p.pages.get(i)
		if err != nil {
			return err
		}

		if p.cache.Contains(page.Id) {
			d, ok := p.cache.Get(page.Id)
			if ok {
				d.(Array).SetAll(input)
			}
		} else {
			err := p.Swap(page.Id)
			if err != nil {
				return err
			}

			d, ok := p.cache.Get(page.Id)
			if ok {
				d.(Array).SetAll(input)
			}
		}

		err = p.flush(page)
		if err != nil {
			return err
		}
	}

	return nil
}

// Data will return nil for a paginated tensor
func (p *Tensor) Data() interface{} {
	return nil
}

// Eq will return if the paginated tensor is equal to
// or not equal to another object.
func (p *Tensor) Eq(i interface{}) bool {
	v, ok := i.(*Tensor)
	if !ok {
		return false
	}
	if v.basePath != p.basePath {
		return false
	}

	if v.dataOrder != p.dataOrder {
		return false
	}

	if v.dtype != p.dtype {
		return false
	}

	return v.dims.Eq(p.dims)
}

// Clone will create a copy of the paginated tensor
// in a subdirectory of the basePath of the original
// paginated tensor.
// It will panic if it encounters an error.
func (p *Tensor) Clone() interface{} {
	name := fmt.Sprintf("%s-%s", "clone", time.Now().Format(stdTimeFmt))
	dir := p.clonePath + name
	err := os.MkdirAll(dir, os.ModePerm)
	if err != nil {
		panic(err)
	}

	clone := &Tensor{
		pages:      make(pages, len(p.pages), len(p.pages)),
		pageSize:   p.pageSize,
		dataOrder:  p.dataOrder,
		dims:       p.dims,
		dtype:      p.dtype,
		transposed: p.transposed,
		engine:     p.engine,
		basePath:   dir + "/",
		clonePath:  dir + "/",
		applyPath:  dir + "/",
		viewPath:   dir + "/",
		fileFormat: p.fileFormat,
	}

	for i := 0; i < len(p.pages); i++ {
		oldPage := p.pages[i]
		newPage := &page{
			Id:     oldPage.Id,
			File:   clone.basePath + strconv.Itoa(oldPage.Id) + "." + p.fileExtension(),
			Bounds: oldPage.Bounds,
		}

		if p.cache.Contains(oldPage.Id) {
			// in-memory
			err := p.flush(oldPage)
			if err != nil {
				panic(err)
			}

			err = p.flush(newPage)
			if err != nil {
				panic(err)
			}

			err = p.load(oldPage)
			if err != nil {
				panic(err)
			}

		} else {
			// not in-memory
			err := p.Swap(oldPage.Id)
			if err != nil {
				panic(err)
			}

			err = p.flush(newPage)
			if err != nil {
				panic(err)
			}

			err = p.load(oldPage)
			if err != nil {
				panic(err)
			}
		}
	}

	return clone
}
