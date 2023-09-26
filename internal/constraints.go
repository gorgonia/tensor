package internal

import "golang.org/x/exp/constraints"

type Addable interface {
	constraints.Float | constraints.Integer | constraints.Complex | ~string
}

type Num interface {
	constraints.Float | constraints.Integer | constraints.Complex
}

type OrderedNum interface {
	constraints.Float | constraints.Integer
}
