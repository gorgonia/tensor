package tensor

import (
	"gorgonia.org/shapes"
)

var xxx Slice = ss(1)
var _ shapes.Slice = xxx

type Slice = shapes.Slice

type rs struct {
	start, end, step int
}

func (s rs) Start() int { return s.start }
func (s rs) End() int   { return s.end }
func (s rs) Step() int  { return s.step }

// makeRS creates a ranged slice. It takes an optional step param.
func makeRS(start, end int, opts ...int) rs {
	step := 1
	if len(opts) > 0 {
		step = opts[0]
	}
	return rs{
		start: start,
		end:   end,
		step:  step,
	}
}

// ss is a single slice, representing this: [start:start+1:0]
type ss int

func (s ss) Start() int { return int(s) }
func (s ss) End() int   { return int(s) + 1 }
func (s ss) Step() int  { return 0 }

// sli is slice. It's named sli to prevent confusion over naming
type sli struct {
	start, end, step int
}

// S creates a Slice.
// end is optional. It should be passed in as the first param of the optionals.
// step is optional. It should be passed in as the second param of the optionals.
//
// Default end is start+1. Default step is 1, unless end == step+1, then it defaults to 0
func S(start int, opt ...int) Slice {
	var end, step int
	if len(opt) > 0 {
		end = opt[0]
	} else {
		end = start + 1
	}

	step = 1
	if len(opt) > 1 {
		step = opt[1]
	} else if end == start+1 {
		step = 0
	}

	return &sli{
		start: start,
		end:   end,
		step:  step,
	}
}

func (s *sli) Start() int { return s.start }
func (s *sli) End() int   { return s.end }
func (s *sli) Step() int  { return s.step }
