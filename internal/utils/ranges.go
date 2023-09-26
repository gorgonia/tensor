package gutils

import (
	"math/rand"
	"time"

	"golang.org/x/exp/constraints"
)

type Rangeable interface {
	constraints.Float | constraints.Integer
}

// Range creates a slice of values of the given DT
func Range[DT Rangeable](start, end int) []DT {
	size := end - start
	incr := true
	if start > end {
		size = start - end
		incr = false
	}
	if size < 0 {
		panic("Cannot create a range that is negative in size")
	}
	retVal := make([]DT, size)
	for i, v := 0, DT(start); i < size; i++ {
		retVal[i] = v
		if incr {
			v++
		} else {
			v--
		}
	}
	return retVal
}

// Random creates a slice of random values of the given DT
func Random[DT Rangeable](size int) []DT {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	retVal := make([]DT, size)
	for i := range retVal {
		retVal[i] = DT(r.Int())
	}
	return retVal

}
