package stdeng

import (
	"golang.org/x/exp/constraints"
	"gorgonia.org/tensor"
)

type OrderedEng[DT constraints.Ordered, T tensor.Basic[DT]] struct {
	StdEng[DT, T]
	compComparableEng[DT, T]
}

// Workhorse returns the engine that will actually do all the work (in this case, itself).
func (e OrderedEng[DT, T]) Workhorse() Engine { return e }
