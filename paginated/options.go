package paginated

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"encoding/json"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"gorgonia.org/tensor"
)

// Load should be the only option used on the tensor if it is called.
// Otherwise, it will overwrite other selected options OR be overwritten
// by other options. In which case, the metadata on the tensor may not actually describe
// the data of the tensor.
func Load(path string) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p, ok := t.(*Tensor)
		if ok {
			mainFile := path + "main.json"
			data, err := ioutil.ReadFile(mainFile)
			if err != nil {
				panic(err)
			}

			err = json.Unmarshal(data, p)
			if err != nil {
				panic(err)
			}

			p.init()

			indexFile := path + "index.gob.gz"
			_, err = os.Stat(indexFile)
			if err == nil {
				index := make(Index)
				f, err := os.Open(indexFile)
				if err != nil {
					panic(err)
				}

				// decompress
				var buf bytes.Buffer
				zr, err := gzip.NewReader(f)
				if err != nil {
					panic(err)
				}
				if _, err := io.Copy(&buf, zr); err != nil {
					panic(err)
				}

				// umarshal
				decoder := gob.NewDecoder(&buf)
				err = decoder.Decode(&index)
				if err != nil {
					panic(err)
				}

				err = p.loadIndex(index)
				if err != nil {
					panic(err)
				}
			}
		}
	}

	return f
}

// WithEngine is an option that can be used to set the engine
// of the tensor to something other than the default `PaginatedEngine`.
func WithEngine(engine tensor.Engine) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p, ok := t.(*Tensor)
		if ok {
			p.engine = engine
		}
	}
	return f
}

// WithName is an option that can be used to set the name
// of the tensor to something other than the default date and time string.
func WithName(name string) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p, ok := t.(*Tensor)
		if ok {
			p.name = name
		}
	}
	return f
}

// DataOrder is an option that can be used to set the data order
// of the tensor to something other than the default RowMajor.
func DataOrder(order tensor.DataOrder) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p, ok := t.(*Tensor)
		if ok {
			p.dataOrder = order
		}
	}
	return f
}

// MaxPageSize specifies the maximum size of a page (in bytes)
// and uses that to estimate the maximum number of values per page.
func MaxPageSize(size int) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p, ok := t.(*Tensor)
		if ok {
			p.pageSize = size / int(p.dtype.Size())
		}
	}
	return f
}

// MaxPageValues specifies the maximum number of values that
// should be in a page.
func MaxPageValues(count int) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p, ok := t.(*Tensor)
		if ok {
			p.pageSize = count
		}
	}
	return f
}

// PageShape is an alternative to specifying the page size
// in values. It simply chunks the tensor using the given shape.
func PageShape(s tensor.Shape) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p, ok := t.(*Tensor)
		if ok {
			p.pageSize = s.TotalSize() / int(p.dtype.Size())
		}
	}
	return f
}

// WithBasePath is a paginated tensor option that
// will set the base path for files for the paginated tensor
func WithBasePath(path string) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p := t.(*Tensor)
		if !strings.HasSuffix(path, "/") {
			path += "/"
		}
		p.basePath = path
	}
	return f
}

// WithClonePath will set the base path for where cloned
// tensors should be stored as sub-directories of.
// This value will default as the base path of the paginated tensor.
func WithClonePath(path string) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p := t.(*Tensor)
		p.clonePath = path
	}
	return f
}

// WithApplyPath will set the base path for where `Apply()` result
// tensors should be stored as sub-directories of.
// This value will default as the base path of the paginated tensor.
func WithApplyPath(path string) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p := t.(*Tensor)
		p.applyPath = path
	}
	return f
}

// WithViewPath will set the base path for where a `view.Materialize()` resulting
// tensor should be stored as a sub-directory of.
// This value will default as the base path of the paginated tensor.
func WithViewPath(path string) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p := t.(*Tensor)
		p.viewPath = path
	}
	return f
}

// WithFileFormat is a PaginatedOption that
// will set the file format for the paginated tensor
func WithFileFormat(format string) tensor.ConsOpt {
	f := func(t tensor.Tensor) {
		p := t.(*Tensor)
		switch format {
		case FormatCSV:
			p.fileFormat = FormatCSV
		case FormatJSON:
			p.fileFormat = FormatJSON
		case FormatProto:
			p.fileFormat = FormatProto
		case FormatFlat:
			p.fileFormat = FormatFlat
		case FormatMsgPck:
			p.fileFormat = FormatMsgPck
		case FormatParquet:
			p.fileFormat = FormatParquet
		case FormatNumpy:
			p.fileFormat = FormatNumpy
		default:
			p.fileFormat = FormatGob
		}
	}
	return f
}

// TODO: WithBackingFile() option:
// -- serialized gorgonia tensor file
// -- serialized gonum matrice file
// -- serialized arrow tensor file
