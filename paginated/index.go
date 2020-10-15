package paginated

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"os"
	"sort"
)

type Index map[interface{}][]int

func (i Index) Save(path string) error {
	var gobBuff, compBuff bytes.Buffer
	zw := gzip.NewWriter(&compBuff)

	filename := path + "index.gob.gz"
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(&gobBuff)
	err = encoder.Encode(i)
	if err != nil {
		return err
	}

	_, err = zw.Write(gobBuff.Bytes())
	if err != nil {
		return err
	}

	_, err = f.Write(compBuff.Bytes())

	return err
}

func (p *Tensor) loadIndex(index Index) error {
	for _, page := range p.pages {
		d, ok := p.cache.Get(page.Id)
		if !ok {
			err := p.Swap(page.Id)
			if err != nil {
				return err
			}

			d, ok = p.cache.Get(page.Id)
		}
		pageData := d.(Array)

		var existing []int
		for value, indices := range index {
			// potential overlap
			if indices[len(indices)-1] >= page.Bounds.S && indices[0] <= page.Bounds.E {
				for _, idx := range indices {
					// within page
					if idx >= page.Bounds.S && idx <= page.Bounds.E {
						err := pageData.SetAt(value, idx)
						if err != nil {
							return err
						}
						existing = append(existing, idx)
					}
				}
			}
		}
		sort.Ints(existing)

		// interpolate the values not-indexed
		var lastValue interface{}
		for i := page.Bounds.S; i <= page.Bounds.E; i++ {
			var exists bool
			for j, e := range existing {
				if e > i {
					break
				}

				if e == i {
					v, err := pageData.At(i - page.Bounds.S)
					if err != nil {
						return err
					}
					lastValue = v

					exists = true
					existing = append(existing[:j], existing[j+1:]...)
					break
				}
			}

			if !exists {
				err := pageData.SetAt(lastValue, i-page.Bounds.S)
				if err != nil {
					return err
				}
			}
		}

		err := p.flush(page)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *Tensor) Index() (Index, error) {
	index := make(Index)

	for _, page := range p.pages {
		d, ok := p.cache.Get(page.Id)
		if !ok {
			err := p.Swap(page.Id)
			if err != nil {
				return nil, err
			}

			d, ok = p.cache.Get(page.Id)
		}
		pageData := d.(Array)

		var lastValue interface{}
		for i := 0; i < pageData.Len(); i++ {
			v, err := pageData.At(i)
			if err != nil {
				return nil, err
			}

			if v == lastValue {
				continue
			}

			indices, ok := index[v]
			if !ok {
				index[v] = []int{i}
			} else {
				indices = append(indices, i)
			}

			lastValue = v
		}
	}

	for _, indices := range index {
		sort.Ints(indices)
	}

	return index, nil
}
