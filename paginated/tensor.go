// paclage paginated provides paginated tensors
package paginated

import (
	"os"
	"runtime"
	"strconv"
	"time"

	"gorgonia.org/tensor"
)

// Tensor specifies a paginated Gorgonia dense tensor
type Tensor struct {
	tensor.Tensor

	name string

	// files
	fileFormat string
	basePath   string
	clonePath  string
	applyPath  string
	viewPath   string

	// tensor
	dataOrder tensor.DataOrder
	dims      tensor.Shape
	dtype     tensor.Dtype
	engine    tensor.Engine

	transposed bool
	old        tensor.Shape

	// pagination
	cache        lru.LRUCache
	memPageCount int // number of pages in memory
	pages        pages
	pageSize     int
}

// New will create a new paginated tensor.
// It will use a `DefaultSmallTensor()` if no options are specified.
func New(dtype tensor.Dtype, opts ...tensor.ConsOpt) (*Tensor, error) {
	/* tensor */
	p, err := DefaultSmallTensor(dtype)
	if err != nil {
		return p, err
	}
	p.engine = NewEngine(p)

	for _, opt := range opts {
		opt(p)
	}

	err = p.init()
	if err != nil {
		return p, err
	}

	return p, nil
}

func (p *Tensor) init() error {
	cache, err := lru.NewLRU(p.memPageCount, nil)
	if err != nil {
		return err
	}
	p.cache = cache

	if p.dims != nil {
		numOfPages := p.Size() / p.pageSize
		if p.Size()%p.pageSize > 0 {
			numOfPages++
		}

		// gen zeroed pages
		for i := 0; i < numOfPages; i++ {
			newPage := &page{
				Id:   len(p.pages),
				File: p.basePath + strconv.Itoa(i) + "." + p.fileExtension(),
				Bounds: Slice{
					S: i * p.pageSize,
					E: (i * p.pageSize) + (p.pageSize - 1),
				},
			}
			p.pages = append(p.pages, newPage)

			// init memory (without consuming too much memory)
			if i == 0 {
				p.cache.Add(0, p.GenArray())
			}

			// write zeroed array as file
			newPage.Id = 0
			err := p.flush(newPage)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// DefaultSmallTensor will consume 1/200 of system memory automatically.
// The returned tensor's memory is not initialized.
func DefaultSmallTensor(dtype tensor.Dtype) (*Tensor, error) {
	m := &runtime.MemStats{}
	runtime.ReadMemStats(m)

	pwd, err := os.Getwd()
	if err != nil {
		return &Tensor{}, err
	}

	t := &Tensor{
		memPageCount: 1,
		pageSize:     int((m.Sys / 200) / uint64(dtype.Size())),
		basePath:     pwd,
		clonePath:    pwd,
		applyPath:    pwd,
		viewPath:     pwd,
		fileFormat:   FormatGob,
		dtype:        dtype,
		pages:        make(pages, 0),
		dataOrder:    0,
		name:         time.Now().Format(stdTimeFmt),
	}

	t.init()

	return t, nil
}

// DefaultMediumTensor will consume 1/20 of system memory automatically
func DefaultMediumTensor(dtype tensor.Dtype) (*Tensor, error) {
	m := &runtime.MemStats{}
	runtime.ReadMemStats(m)

	pwd, err := os.Getwd()
	if err != nil {
		return &Tensor{}, err
	}

	t := &Tensor{
		memPageCount: 10,
		pageSize:     int((m.Sys / 200) / uint64(dtype.Size())),
		basePath:     pwd,
		clonePath:    pwd,
		applyPath:    pwd,
		viewPath:     pwd,
		fileFormat:   FormatGob,
		dtype:        dtype,
		pages:        make(pages, 0),
		dataOrder:    0,
		name:         time.Now().Format(stdTimeFmt),
	}

	t.init()

	return t, nil
}

// DefaultLargeTensor will consume 1/2 of system memory automatically
func DefaultLargeTensor(dtype tensor.Dtype) (*Tensor, error) {
	m := &runtime.MemStats{}
	runtime.ReadMemStats(m)

	pwd, err := os.Getwd()
	if err != nil {
		return &Tensor{}, err
	}

	t := &Tensor{
		memPageCount: 100,
		pageSize:     int((m.Sys / 200) / uint64(dtype.Size())),
		basePath:     pwd,
		clonePath:    pwd,
		applyPath:    pwd,
		viewPath:     pwd,
		fileFormat:   FormatGob,
		dtype:        dtype,
		pages:        make(pages, 0),
		dataOrder:    0,
		name:         time.Now().Format(stdTimeFmt),
	}

	t.init()

	return t, nil
}

func (p *Tensor) largestCoord() []int {
	coord := make([]int, p.Dims(), p.Dims())
	for i, dim := range p.Shape() {
		coord[i] = dim - 1
	}

	return coord
}

// pageIndex will return the index of the coordinate
// inside it's containing page
func (p *Tensor) pageIndex(coords ...int) int {
	var index int
	for i, coord := range coords {
		switch i {
		case 0:
			index += coord * p.rowSize()
		case 1:
			index += coord * p.columnSize()
		default:
			index += coord * p.dimensionSize(i)
		}
	}

	return index % p.pageSize
}

// globalIndex
func (p *Tensor) globalIndex(coords ...int) int {
	var index int
	for i, coord := range coords {
		switch i {
		case 0:
			index += coord * p.rowSize()
		case 1:
			index += coord * p.columnSize()
		default:
			index += coord * p.dimensionSize(i)
		}
	}

	return index
}

// page will return a pointer to the page which contains
// the value at the given coords
func (p *Tensor) page(coords ...int) (*page, error) {
	var sum int
	for i, coord := range coords {
		switch i {
		case 0:
			sum += coord * p.rowSize()
		case 1:
			sum += coord * p.columnSize()
		default:
			sum += coord * p.dimensionSize(i)
		}
	}

	pageId := sum / p.pageSize
	if pageId > len(p.pages)-1 {
		return &page{}, ErrDNE
	}

	return p.pages[pageId], nil
}

// rowSize is number of columns if row-major
// and is 1 if column-major
// 0th-dimension
func (p *Tensor) rowSize() int {
	switch p.DataOrder() {
	case 1:
		return 1
	default:
		if p.Dims() > 1 {
			return p.Shape()[1]
		}

		return p.Shape()[0]
	}
}

// column-size is number of rows if column-major
// and is 1 if row-major
// 1st dimension
func (p *Tensor) columnSize() int {
	if p.DataOrder() == 1 && p.Dims() > 1 {
		return p.Shape()[0]
	}

	return 0
}

// will return size of dimensions beyond the 0th (1st) and 1st (2nd) dimensions
func (p *Tensor) dimensionSize(i int) int {
	if len(p.Shape()) <= i {
		return 0
	}

	var size int
	for dim := 0; dim < i; dim++ {
		size *= p.Shape()[dim]
	}

	return size
}

func (p *Tensor) previousCoord(coords []int) []int {
	switch p.DataOrder() {
	case 1:
		coords[1] = p.decrementCoord(1, coords[1])
		if coords[1] == p.Shape()[1]-1 {
			coords[0] = p.decrementCoord(0, coords[0])
		}

		if p.Dims() > 2 {
			upperIndex := p.Dims() - 1
			for i := 0; i < (upperIndex - 1); i-- {
				index := upperIndex - i
				if index == upperIndex {
					coords[index] = p.decrementCoord(index, coords[index])
				} else if coords[index+1] == p.Shape()[index]-1 {
					coords[index] = p.decrementCoord(index, coords[index])
				}
			}
		}
	default:
		upperIndex := p.Dims() - 1
		for i := 0; i < (upperIndex - 1); i-- {
			index := upperIndex - i
			if index == upperIndex {
				coords[index] = p.decrementCoord(index, coords[index])
			} else if coords[i+1] == p.Shape()[index]-1 {
				coords[index] = p.decrementCoord(index, coords[index])
			}
		}
	}

	return coords
}

func (p *Tensor) nextCoord(coords []int) []int {
	switch p.DataOrder() {
	case 1:
		coords[1] = p.incrementCoord(1, coords[1])
		if coords[1] == 0 {
			coords[0] = p.incrementCoord(0, coords[0])
		}

		if p.Dims() > 2 {
			for i := range coords[2:] {
				if coords[i-1] == 0 {
					coords[i] = p.incrementCoord(i, coords[i])
				}
			}
		}
	default:
		for i := range coords {
			if i == 0 {
				coords[i] = p.incrementCoord(i, coords[i])
			} else {
				if coords[i-1] == 0 {
					coords[i] = p.incrementCoord(i, coords[i])
				}
			}
		}
	}

	return coords
}

func (p *Tensor) decrementCoord(index, value int) int {
	if value > 0 {
		return value - 1
	}

	return p.Shape()[index] - 1
}

func (p *Tensor) incrementCoord(index, value int) int {
	if value < (p.Shape()[index] - 1) {
		return value + 1
	}

	return 0
}
