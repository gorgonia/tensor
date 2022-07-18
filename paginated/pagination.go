package paginated

import (
	"encoding/csv"
	"encoding/gob"
	"encoding/json"
	"io/ioutil"
	"os"

	"gorgonia.org/tensor"
)

type page struct {
	Id       int
	File     string
	Datasize int // number of values written to page; (not total page capacity)
	Bounds   Slice
}

type pages []*page

func (p pages) get(id int) (*page, error) {
	for _, page := range p {
		if page.Id == id {
			return page, nil
		}
	}

	return &page{}, ErrDNE
}

// Swap will load the desired page's data into
// the position of the least-used page in memory.
func (p *Tensor) Swap(id int) error {
	page, err := p.pages.get(id)
	if err != nil {
		return err
	}

	// if buffer can grow: load page; no swap
	if p.cache.Len() < p.memPageCount {
		p.cache.Add(id, p.GenArray())
		return p.load(page)
	}

	// else: swap with oldest page
	k, v, _ := p.cache.GetOldest()
	oldPage, err := p.pages.get(k.(int))
	if err != nil {
		return err
	}

	err = p.flush(oldPage)
	if err != nil {
		return err
	}

	v.(Array).Zero()
	p.cache.Add(id, v)

	return p.load(page)
}

func (p *Tensor) intersectingPages(slices ...tensor.Slice) pages {
	var pages pages

	minCoord := make([]int, p.Dims(), p.Dims())
	maxCoord := make([]int, p.Dims(), p.Dims())
	for i, s := range slices {
		minCoord[i] = s.Start()
		maxCoord[i] = s.End()
	}

	b := Slice{
		S: p.globalIndex(minCoord...),
		E: p.globalIndex(maxCoord...),
	}

	for _, page := range p.pages {
		if page.Bounds.intersects(b) {
			pages = append(pages, page)
		}
	}

	return pages
}

func (p *Tensor) flush(page *page) error {
	switch p.fileFormat {
	case FormatCSV:
		f, err := os.OpenFile(page.File, os.O_WRONLY|os.O_CREATE, 0666)
		if err != nil {
			return err
		}

		w := csv.NewWriter(f)
		d, ok := p.cache.Get(page.Id)
		if ok {
			strings, err := toStrings(p.Dtype(), d.(Array))
			if err != nil {
				return err
			}

			err = w.Write(strings)
			if err != nil {
				return err
			}
		}

		return f.Close()
	case FormatJSON:
		d, ok := p.cache.Get(page.Id)
		if !ok {
			return ErrDNE
		}

		data, err := json.Marshal(d.(Array))
		if err != nil {
			return err
		}

		return ioutil.WriteFile(page.File, data, 0644)
	case FormatGob:
		f, err := os.OpenFile(page.File, os.O_WRONLY|os.O_CREATE, 0666)
		if err != nil {
			return err
		}

		encoder := gob.NewEncoder(f)
		d, ok := p.cache.Get(page.Id)
		if ok {
			err = encoder.Encode(d.(Array))
			if err != nil {
				return err
			}
		}

		return f.Close()
	case FormatProto:
		// TODO: implement
	case FormatFlat:
		// TODO: implement
	case FormatMsgPck:
		// TODO: implement
	case FormatParquet:
		// TODO: implement
	case FormatNumpy:
		// TODO: implement
	default:
		return ErrFormat
	}

	return nil
}

func (p *Tensor) load(page *page) error {
	switch p.fileFormat {
	case FormatCSV:
		f, err := os.Open(page.File)
		if err != nil {
			return err
		}
		r := csv.NewReader(f)

		// always one line
		data, err := r.Read()
		if err != nil {
			return err
		}

		d, ok := p.cache.Get(page.Id)
		if !ok {
			return ErrDNE
		}
		a := d.(Array)
		for i := 0; i < len(data); i++ {
			err := a.SetAt(parseString(p.Dtype(), data[i]), i)
			if err != nil {
				return err
			}
		}
	case FormatJSON:
		data, err := ioutil.ReadFile(page.File)
		if err != nil {
			return err
		}

		d, ok := p.cache.Get(page.Id)
		if !ok {
			return ErrDNE
		}

		return json.Unmarshal(data, d)
	case FormatGob:
		f, err := os.Open(page.File)
		if err != nil {
			return err
		}

		decoder := gob.NewDecoder(f)

		d, ok := p.cache.Get(page.Id)
		if !ok {
			return ErrDNE
		}

		err = decoder.Decode(d)
		if err != nil {
			return err
		}

		return f.Close()
	case FormatProto:
		// TODO: implement
	case FormatFlat:
		// TODO: implement
	case FormatMsgPck:
		// TODO: implement
	case FormatParquet:
		// TODO: implement
	case FormatNumpy:
		// TODO: implement
	default:
		return ErrFormat
	}

	return nil
}

func (p *Tensor) firstUnfilledPage() (*page, bool) {
	for _, page := range p.pages {
		if page.Datasize < p.pageSize {
			return page, true
		}
	}

	return nil, false
}
