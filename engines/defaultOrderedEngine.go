package stdeng

import (
	"gorgonia.org/tensor"
	"golang.org/x/exp/constraints"
)

type OrderedEng[DT constraints.Ordered, T tensor.Basic[DT]] struct {
	StdEng[DT, T]
	compComparableEng[DT, T]
}
