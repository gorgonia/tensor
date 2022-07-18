package paginated

import (
	"fmt"
	"os"
	"time"

	"gorgonia.org/tensor"
)

// View specifies a view of a
// paginated tensor. It can be materialized
// into a new paginated tensor if desired.
type View struct {
	*Tensor
	name   string
	slices []tensor.Slice
}

// NewView will create and return a new View object.
func NewView(p *Tensor, slices ...tensor.Slice) *View {
	return &View{
		Tensor: p,
		name:   "view",
		slices: slices,
	}
}

// SetName will set the name of the view that will be used
func (p *View) SetName(name string) {
	p.name = name
}

// IsView will specify if object is only a view
// of another tensor or if it is it's own tensor.
func (p *View) IsView() bool {
	return true
}

// IsMaterializable specifies if the view can be converted
// into it's own tensor. This always returns true for a `*View`
func (p *View) IsMaterializable() bool {
	return true
}

// Materialize will convert the `*View` into a new `*Tensor`.
func (p *View) Materialize() tensor.Tensor {
	intersectingPages := p.intersectingPages(p.slices...)

	b := p.sliceCoords(p.slices...)
	dataSize := p.boundsData(b)
	numOfPages := dataSize / p.pageSize
	if dataSize%p.pageSize > 0 {
		numOfPages++
	}
	globalStart := p.pageIndex(b.min...)
	globalEnd := p.pageIndex(b.max...)

	// view
	name := fmt.Sprintf("%s-%s", p.name, time.Now().Format(stdTimeFmt))
	dir := p.viewPath + name
	err := os.MkdirAll(dir, os.ModePerm)
	if err != nil {
		panic(err)
	}

	dims := make(tensor.Shape, len(p.slices), len(p.slices))
	for i := 0; i < len(dims); i++ {
		if i < len(dims)-1 {
			dims[i] = p.dims[i]
			continue
		}

		dims[i] = (b.max[i] - b.min[i]) + 1
	}

	view := &Tensor{
		pages:        make(pages, numOfPages, numOfPages),
		pageSize:     p.pageSize,
		memPageCount: numOfPages,
		dataOrder:    p.dataOrder,
		dims:         dims,
		dtype:        p.dtype,
		transposed:   false,
		engine:       p.engine,
		basePath:     dir + "/",
		fileFormat:   p.fileFormat,
	}

	err = view.init()
	if err != nil {
		panic(err)
	}

	// variables updated every loop
	var (
		to   = 0
		from = 0

		fromProcessed = 0
		toFilled      = 0

		remainingData = dataSize
	)

	// copy loop
	for {
		toPage := view.pages[to]
		toCapacity := view.pageSize - toPage.Datasize
		toCapacity -= toFilled

		fromPage := intersectingPages[from]
		fromStart := p.pageSize * fromPage.Id
		fromEnd := fromStart + p.pageSize
		if fromStart < globalStart {
			fromStart = globalStart
		}
		if fromEnd > globalEnd {
			fromEnd = globalEnd
		}
		fromStart += fromProcessed
		fromRemaining := fromEnd - fromStart

		// to-data
		td, ok := view.cache.Get(toPage.Id)
		if !ok {
			err := view.Swap(toPage.Id)
			if err != nil {
				panic(err)
			}

			td, ok = view.cache.Get(toPage.Id)
			if !ok {
				panic(ErrCache)
			}
		}
		toData := td.(Array)

		// from-data
		fd, ok := p.cache.Get(fromPage.Id)
		if !ok {
			err := p.Swap(fromPage.Id)
			if err != nil {
				panic(err)
			}

			fd, ok = p.cache.Get(fromPage.Id)
			if !ok {
				panic(ErrCache)
			}
		}
		fromData := fd.(Array)

		// slice
		capacity := min(toCapacity, fromRemaining)
		toSlice, err := toData.Slice(toPage.Datasize, capacity)
		if err != nil {
			panic(err)
		}
		fromSlice, err := fromData.Slice(fromStart, capacity)
		if err != nil {
			panic(err)
		}

		// copy
		err = toSlice.Copy(fromSlice)
		if err != nil {
			panic(err)
		}
		toPage.Datasize += capacity

		// update
		if toCapacity < fromRemaining {
			err := view.flush(toPage)
			if err != nil {
				panic(err)
			}

			fromProcessed += capacity

			toFilled = 0
			to++

			remainingData -= capacity
		} else if fromRemaining < toCapacity {
			fromProcessed = 0
			from++

			toFilled += capacity

			remainingData -= capacity
		} else {
			err := view.flush(toPage)
			if err != nil {
				panic(err)
			}

			fromProcessed = 0
			from++

			toFilled = 0
			to++

			remainingData -= capacity
		}

		// exit
		if from == len(intersectingPages) && remainingData <= 0 {
			break
		} else if from == len(intersectingPages) && remainingData >= 0 {
			panic("something went wrong materializing the view")
		}
	}

	return view
}

func min(arg1, arg2 int) int {
	if arg1 < arg2 {
		return arg1
	}

	return arg2
}

type bounds struct {
	min, max []int
}

// slicecoords can be used on a set of slices that is assumed to be
// a slice along each dimension of the tensor defining a "chunk" of the tensor.
func (p *Tensor) sliceCoords(slices ...tensor.Slice) bounds {
	minCoord := make([]int, p.Dims(), p.Dims())
	maxCoord := make([]int, p.Dims(), p.Dims())
	for i, s := range slices {
		minCoord[i] = s.Start()
		maxCoord[i] = s.End()
	}

	return bounds{
		min: minCoord,
		max: maxCoord,
	}
}

func (p *Tensor) boundsData(b bounds) int {
	start := p.globalIndex(b.min...)
	end := p.globalIndex(b.max...)
	return end - start
}
