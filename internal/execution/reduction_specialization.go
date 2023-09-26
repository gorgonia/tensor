package execution

import (
	"golang.org/x/exp/constraints"
)

func MonotonicMax[T constraints.Ordered](a []T) T {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	retVal := a[0]
	for i := range a {
		if a[i] > retVal {
			retVal = a[i]
		}
	}
	return retVal
}

func Max0[T constraints.Ordered](a, b []T) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func Max[T constraints.Ordered](a []T, defVal T) T {
	retVal := defVal
	for i := range a {
		if a[i] > retVal {
			retVal = a[i]
		}
	}
	return retVal
}

func MonotonicSum[T Num](a []T) T {
	var retVal T
	if len(a) < 1 {
		return retVal
	}
	for i := range a {
		retVal += a[i]
	}
	return retVal
}

func Sum0[T Num](a, b []T) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v + b[i]
	}
}

func Sum[T Num](a []T, defVal T) T {
	retVal := defVal
	for i := range a {
		retVal += a[i]
	}
	return retVal
}

func MonotonicProd[T Num](a []T) T {
	var retVal T
	if len(a) < 1 {
		return retVal
	}
	retVal = a[0]
	for i := range a {
		retVal *= a[i]
	}
	return retVal
}

func Prod0[T Num](a, b []T) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v * b[i]
	}
}

func Prod[T Num](a []T, defVal T) T {
	retVal := defVal
	for i := range a {
		retVal *= a[i]
	}
	return retVal
}
