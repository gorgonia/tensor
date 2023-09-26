package execution

import (
	"golang.org/x/exp/constraints"
)

type Addable interface {
	constraints.Ordered | constraints.Complex
}

type Num interface {
	constraints.Float | constraints.Integer | constraints.Complex
}

type OrderedNum interface {
	constraints.Float | constraints.Integer
}
