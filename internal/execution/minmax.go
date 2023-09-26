package execution

import (
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

func MinVV[T OrderedNum](a, b, c []T) {
	a = a[:]
	b = b[:len(a)]
	c = c[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			c[i] = bv
		} else {
			c[i] = v
		}
	}
}

func MinVS[T OrderedNum](a []T, b T, c []T) {
	a = a[:]
	c = c[:len(a)]
	for i, v := range a {
		if b < v {
			c[i] = b
		} else {
			c[i] = v
		}
	}
}

func MinSV[T OrderedNum](a T, b []T, c []T) {
	b = b[:]
	c = c[:len(b)]
	for i, v := range b {
		if a < v {
			c[i] = a
		} else {
			c[i] = v
		}
	}
}

func MaxVV[T OrderedNum](a, b, c []T) {
	a = a[:]
	b = b[:len(a)]
	c = c[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			c[i] = bv
		} else {
			c[i] = v
		}
	}
}

func MaxVS[T OrderedNum](a []T, b T, c []T) {
	a = a[:]
	c = c[:len(a)]
	for i, v := range a {
		if b > v {
			c[i] = b
		} else {
			c[i] = v
		}
	}
}

func MaxSV[T OrderedNum](a T, b []T, c []T) {
	b = b[:]
	c = c[:len(b)]
	for i, v := range b {
		if a > v {
			c[i] = a
		} else {
			c[i] = v
		}
	}
}

func MinVVIter[T OrderedNum](a, b, c []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool

	if cisa, cisb, err = iterCheck3(a, b, c, ait, bit, cit); err != nil {
		return errors.Wrapf(err, internal.ThisFn())
	}
loop:
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}

		switch {
		case cisa:
			validk = validi
			k = i
		case cisb:
			validk = validj
			k = j
		default:
			if k, validk, err = cit.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj && validk {
			if a[i] < b[j] {
				c[k] = a[i]
			} else {
				c[k] = b[j]
			}
		}
	}
	return
}

func MaxVVIter[T OrderedNum](a, b, c []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool

	if cisa, cisb, err = iterCheck3(a, b, c, ait, bit, cit); err != nil {
		return errors.Wrapf(err, internal.ThisFn())
	}
loop:
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}

		switch {
		case cisa:
			validk = validi
			k = i
		case cisb:
			validk = validj
			k = j
		default:
			if k, validk, err = cit.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj && validk {
			if a[i] > b[j] {
				c[k] = a[i]
			} else {
				c[k] = b[j]
			}
		}
	}
	return
}

func MinSVIter[T OrderedNum](s T, vec, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "AddVSIter")
	}
loop:
	for {
		if i, validi, err = veciter.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if retIsVec {
			j = i
			validj = validi
		} else {
			if j, validj, err = retiter.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj {
			if s < vec[i] {
				retVal[j] = s
			} else {
				retVal[j] = vec[i]
			}
		}
	}
	return
}

func MinVSIter[T OrderedNum](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "AddVSIter")
	}
loop:
	for {
		if i, validi, err = veciter.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if retIsVec {
			j = i
			validj = validi
		} else {
			if j, validj, err = retiter.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj {
			if vec[i] < s {
				retVal[j] = vec[i]
			} else {
				retVal[j] = s
			}
		}
	}
	return
}

func MaxSVIter[T OrderedNum](s T, vec, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "AddVSIter")
	}
loop:
	for {
		if i, validi, err = veciter.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if retIsVec {
			j = i
			validj = validi
		} else {
			if j, validj, err = retiter.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj {
			if s > vec[i] {
				retVal[j] = s
			} else {
				retVal[j] = vec[i]
			}
		}
	}
	return
}

func MaxVSIter[T OrderedNum](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "AddVSIter")
	}
loop:
	for {
		if i, validi, err = veciter.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if retIsVec {
			j = i
			validj = validi
		} else {
			if j, validj, err = retiter.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj {
			if vec[i] > s {
				retVal[j] = vec[i]
			} else {
				retVal[j] = s
			}
		}
	}
	return
}
