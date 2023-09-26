package execution

import (
	"gorgonia.org/tensor/internal"
	"golang.org/x/exp/constraints"
)

func Argmax[T constraints.Ordered](a []T) int {
	var set bool
	var f T
	var max int
	for i := range a {
		v := a[i]
		if !set {
			f = v
			max = i
			set = true
			continue
		}
		if v > f {
			max = i
			f = v
		}
	}
	return max
}

func ArgmaxIter[T constraints.Ordered](a []T, it Iterator, lastSize int) (indices []int, err error) {
	tmp := make([]T, 0, lastSize)
	for next, err := it.Next(); err == nil; next, err = it.Next() {
		tmp = append(tmp, a[next])
		if len(tmp) == lastSize {
			am := Argmax(tmp)
			indices = append(indices, am)

			// reset
			tmp = tmp[:0]
		}
	}

	err = internal.HandleNoOp(err)
	return

}

func Argmin[T constraints.Ordered](a []T) int {
	var set bool
	var f T
	var min int
	for i := range a {
		v := a[i]
		if !set {
			f = v
			min = i
			set = true
			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func ArgminIter[T constraints.Ordered](a []T, it Iterator, lastSize int) (indices []int, err error) {
	tmp := make([]T, 0, lastSize)
	for next, err := it.Next(); err == nil; next, err = it.Next() {
		tmp = append(tmp, a[next])
		if len(tmp) == lastSize {
			am := Argmin(tmp)
			indices = append(indices, am)

			// reset
			tmp = tmp[:0]
		}
	}

	err = internal.HandleNoOp(err)
	return

}
