package paginated

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math/bits"
	"strconv"

	"gorgonia.org/tensor"
)

// Save will save the tensor metadata in a main.json file in the
// basepath (defaults to pwd) of the tensor.
func (p *Tensor) Save() error {
	filename := p.basePath + "main.json"

	data, err := p.MarshalJSON()
	if err != nil {
		return err
	}

	return ioutil.WriteFile(filename, data, 0666)
}

// Flush will write all the data in-memory
// to disk.
func (p *Tensor) Flush() error {
	for _, page := range p.pages {
		if p.cache.Contains(page.Id) {
			err := p.flush(page)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// WriteNpy is not implemented.
// It will return ErrNotImplemented.
func (p *Tensor) WriteNpy(w io.Writer) error {
	return ErrNotImplemented
}

// ReadNpy will load the data from a numpy object
// into the tensor.
func (p *Tensor) ReadNpy(r io.Reader) error {
	return nil
}

// GobDecode is not implemented
// It will return ErrNotImplemented
func (p *Tensor) GobDecode(data []byte) error {
	return ErrNotImplemented
}

// GobEncode is not implemented
// It will return ErrNotImplemented
func (p *Tensor) GobEncode() ([]byte, error) {
	return []byte{}, ErrNotImplemented
}

func (p *Tensor) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Name         string
		FileFormat   string
		BasePath     string
		ClonePath    string
		ApplyPath    string
		ViewPath     string
		DataOrder    tensor.DataOrder
		Dims         tensor.Shape
		DType        tensor.Dtype
		Engine       tensor.Engine
		Transposed   bool
		Old          tensor.Shape
		MemPageCount int
		Pages        pages
		PageSize     int
	}{
		Name:         p.name,
		FileFormat:   p.fileFormat,
		BasePath:     p.basePath,
		ClonePath:    p.clonePath,
		ApplyPath:    p.applyPath,
		ViewPath:     p.viewPath,
		DataOrder:    p.dataOrder,
		Dims:         p.dims,
		DType:        p.dtype,
		Engine:       p.engine,
		Transposed:   p.transposed,
		Old:          p.old,
		MemPageCount: p.memPageCount,
		Pages:        p.pages,
		PageSize:     p.pageSize,
	})
}

func (p *Tensor) UnmarshalJSON(data []byte) error {
	type tnsr struct {
		Name         string
		FileFormat   string
		BasePath     string
		ClonePath    string
		ApplyPath    string
		ViewPath     string
		DataOrder    tensor.DataOrder
		Dims         tensor.Shape
		DType        tensor.Dtype
		Engine       tensor.Engine
		Transposed   bool
		Old          tensor.Shape
		MemPageCount int
		Pages        pages
		PageSize     int
	}

	var t tnsr
	if err := json.Unmarshal(data, &t); err != nil {
		return err
	}

	p.name = t.Name
	p.fileFormat = t.FileFormat
	p.basePath = t.BasePath
	p.clonePath = t.ClonePath
	p.applyPath = t.ApplyPath
	p.viewPath = t.ViewPath
	p.dataOrder = t.DataOrder
	p.dims = t.Dims
	p.dtype = t.DType
	p.engine = t.Engine
	p.transposed = t.Transposed
	p.old = t.Old
	p.memPageCount = t.MemPageCount
	p.pages = t.Pages
	p.pageSize = t.PageSize

	return nil
}

// toStrings will convert an underlying data slice to a slice of strings
// primarily used to write a page's data to a CSV file
func toStrings(dtype tensor.Dtype, a Array) ([]string, error) {
	length := a.Len()
	strings := make([]string, length, length)
	for i := 0; i < length; i++ {
		v, err := a.At(i)
		if err != nil {
			return strings, err
		}
		strings[i] = fmt.Sprintf("%v", v)
	}

	return strings, nil
}

func parseString(dtype tensor.Dtype, s string) interface{} {
	switch dtype {
	case tensor.Bool:
		b, _ := strconv.ParseBool(s)
		return b
	case tensor.Int:
		var i int64
		if is64Bit {
			i, _ = strconv.ParseInt(s, 10, 64)
		} else {
			i, _ = strconv.ParseInt(s, 10, 32)
		}

		return int(i)
	case tensor.Int8:
		i, _ := strconv.ParseInt(s, 10, 8)
		return i
	case tensor.Int16:
		i, _ := strconv.ParseInt(s, 10, 16)
		return i
	case tensor.Int32:
		i, _ := strconv.ParseInt(s, 10, 32)
		return i
	case tensor.Int64:
		i, _ := strconv.ParseInt(s, 10, 64)
		return i
	case tensor.Uint:
		u, _ := strconv.ParseUint(s, 10, bits.UintSize)
		return u
	case tensor.Uint8, tensor.Byte:
		u, _ := strconv.ParseUint(s, 10, 8)
		return u
	case tensor.Uint16:
		u, _ := strconv.ParseUint(s, 10, 16)
		return u
	case tensor.Uint32:
		u, _ := strconv.ParseUint(s, 10, 32)
		return u
	case tensor.Uint64:
		u, _ := strconv.ParseUint(s, 10, 64)
		return u
	case tensor.Float32:
		f, _ := strconv.ParseFloat(s, 32)
		return f
	case tensor.Float64:
		f, _ := strconv.ParseFloat(s, 64)
		return f
	case tensor.Complex64:
		c, _ := strconv.ParseComplex(s, 64)
		return c
	case tensor.Complex128:
		c, _ := strconv.ParseComplex(s, 128)
		return c
	case tensor.String:
		return s
	}

	return nil
}
