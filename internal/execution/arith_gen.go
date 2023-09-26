// Code generated by genlib3. DO NOT EDIT

package execution

import "gorgonia.org/tensor/internal/errors"

// AddVV does c :=  a ̅+ b
func AddVV[T Addable](a, b, c []T) {
	a = a[:]
	b = b[:len(a)]
	c = c[:len(a)]

	for i := range a {
		c[i] = a[i] + b[i]
	}
}

// AddVS does c := vec ̅+ scalar. The scalar value is broadcasted across the vector for the operation
func AddVS[T Addable](vec []T, s T, retVal []T) {
	vec = vec[:]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = vec[i] + s
	}
}

// AddSV does c := scalar ̅+ vector. The scalar value is broadcasted across the vector for the operation.
func AddSV[T Addable](s T, vec []T, retVal []T) {
	vec = vec[:]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = s + vec[i]
	}
}

// AddVVIncr does c += a ̅+ b
func AddVVIncr[T Addable](a, b, incr []T) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]

	for i := range a {
		incr[i] += a[i] + b[i]
	}
}

// AddVVIter does c := a ̅+ b, where a, b, and c requires the use of an iterator.
func AddVVIter[T Addable](a, b, c []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, c, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "AddVVIter")
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
			c[k] = a[i] + b[j]
		}
	}
	return

}

// AddVVIncrIter does c += a ̅+ b, where a, b, and c requires the use of an iterator.
func AddVVIncrIter[T Addable](a, b, incr []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, incr, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "AddVVIncrIter")
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
			incr[k] += a[i] + b[j]
		}
	}
	return
}

// AddVSIncr performs c += vec ̅+ scalar. The scalar value is broadcasted across the vector for the operation.
func AddVSIncr[T Addable](vec []T, s T, incr []T) {
	vec = vec[:]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += vec[i] + s
	}
}

// AddSVIncr performs c += scalar ̅+ vector. The scalar value is broadcasted across the vector for the operation.
func AddSVIncr[T Addable](s T, vec []T, incr []T) {
	vec = vec[:]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += s + vec[i]
	}
}

// AddVSIter performs c := vec ̅+ scalar, where vec and c both require iterators.
func AddVSIter[T Addable](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
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
			retVal[j] = vec[i] + s
		}
	}
	return
}

// AddSVIter performs c := scalar ̅+ vector, where vec and c both require iterators.
func AddSVIter[T Addable](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
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
			retVal[j] = s + vec[i]
		}
	}
	return
}

// AddVSIncrIter performs c += vec ̅+ scalar, where vec and c both require iterators.
func AddVSIncrIter[T Addable](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "AddVSIncrIter")
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
			retVal[j] += vec[i] + s
		}
	}
	return
}

// AddSVIncrIter performs c += scalar ̅+ vector, where vec and c both require iterators.
func AddSVIncrIter[T Addable](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "AddVSIncrIter")
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
			retVal[j] += s + vec[i]
		}
	}
	return
}

// SubVV does c :=  a ̅- b
func SubVV[T Num](a, b, c []T) {
	a = a[:]
	b = b[:len(a)]
	c = c[:len(a)]

	for i := range a {
		c[i] = a[i] - b[i]
	}
}

// SubVS does c := vec ̅- scalar. The scalar value is broadcasted across the vector for the operation
func SubVS[T Num](vec []T, s T, retVal []T) {
	vec = vec[:]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = vec[i] - s
	}
}

// SubSV does c := scalar ̅- vector. The scalar value is broadcasted across the vector for the operation.
func SubSV[T Num](s T, vec []T, retVal []T) {
	vec = vec[:]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = s - vec[i]
	}
}

// SubVVIncr does c += a ̅- b
func SubVVIncr[T Num](a, b, incr []T) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]

	for i := range a {
		incr[i] += a[i] - b[i]
	}
}

// SubVVIter does c := a ̅- b, where a, b, and c requires the use of an iterator.
func SubVVIter[T Num](a, b, c []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, c, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "SubVVIter")
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
			c[k] = a[i] - b[j]
		}
	}
	return

}

// SubVVIncrIter does c += a ̅- b, where a, b, and c requires the use of an iterator.
func SubVVIncrIter[T Num](a, b, incr []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, incr, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "SubVVIncrIter")
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
			incr[k] += a[i] - b[j]
		}
	}
	return
}

// SubVSIncr performs c += vec ̅- scalar. The scalar value is broadcasted across the vector for the operation.
func SubVSIncr[T Num](vec []T, s T, incr []T) {
	vec = vec[:]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += vec[i] - s
	}
}

// SubSVIncr performs c += scalar ̅- vector. The scalar value is broadcasted across the vector for the operation.
func SubSVIncr[T Num](s T, vec []T, incr []T) {
	vec = vec[:]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += s - vec[i]
	}
}

// SubVSIter performs c := vec ̅- scalar, where vec and c both require iterators.
func SubVSIter[T Num](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "SubVSIter")
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
			retVal[j] = vec[i] - s
		}
	}
	return
}

// SubSVIter performs c := scalar ̅- vector, where vec and c both require iterators.
func SubSVIter[T Num](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "SubVSIter")
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
			retVal[j] = s - vec[i]
		}
	}
	return
}

// SubVSIncrIter performs c += vec ̅- scalar, where vec and c both require iterators.
func SubVSIncrIter[T Num](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "SubVSIncrIter")
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
			retVal[j] += vec[i] - s
		}
	}
	return
}

// SubSVIncrIter performs c += scalar ̅- vector, where vec and c both require iterators.
func SubSVIncrIter[T Num](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "SubVSIncrIter")
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
			retVal[j] += s - vec[i]
		}
	}
	return
}

// MulVV does c :=  a ̅* b
func MulVV[T Num](a, b, c []T) {
	a = a[:]
	b = b[:len(a)]
	c = c[:len(a)]

	for i := range a {
		c[i] = a[i] * b[i]
	}
}

// MulVS does c := vec ̅* scalar. The scalar value is broadcasted across the vector for the operation
func MulVS[T Num](vec []T, s T, retVal []T) {
	vec = vec[:]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = vec[i] * s
	}
}

// MulSV does c := scalar ̅* vector. The scalar value is broadcasted across the vector for the operation.
func MulSV[T Num](s T, vec []T, retVal []T) {
	vec = vec[:]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = s * vec[i]
	}
}

// MulVVIncr does c += a ̅* b
func MulVVIncr[T Num](a, b, incr []T) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]

	for i := range a {
		incr[i] += a[i] * b[i]
	}
}

// MulVVIter does c := a ̅* b, where a, b, and c requires the use of an iterator.
func MulVVIter[T Num](a, b, c []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, c, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "MulVVIter")
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
			c[k] = a[i] * b[j]
		}
	}
	return

}

// MulVVIncrIter does c += a ̅* b, where a, b, and c requires the use of an iterator.
func MulVVIncrIter[T Num](a, b, incr []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, incr, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "MulVVIncrIter")
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
			incr[k] += a[i] * b[j]
		}
	}
	return
}

// MulVSIncr performs c += vec ̅* scalar. The scalar value is broadcasted across the vector for the operation.
func MulVSIncr[T Num](vec []T, s T, incr []T) {
	vec = vec[:]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += vec[i] * s
	}
}

// MulSVIncr performs c += scalar ̅* vector. The scalar value is broadcasted across the vector for the operation.
func MulSVIncr[T Num](s T, vec []T, incr []T) {
	vec = vec[:]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += s * vec[i]
	}
}

// MulVSIter performs c := vec ̅* scalar, where vec and c both require iterators.
func MulVSIter[T Num](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "MulVSIter")
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
			retVal[j] = vec[i] * s
		}
	}
	return
}

// MulSVIter performs c := scalar ̅* vector, where vec and c both require iterators.
func MulSVIter[T Num](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "MulVSIter")
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
			retVal[j] = s * vec[i]
		}
	}
	return
}

// MulVSIncrIter performs c += vec ̅* scalar, where vec and c both require iterators.
func MulVSIncrIter[T Num](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "MulVSIncrIter")
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
			retVal[j] += vec[i] * s
		}
	}
	return
}

// MulSVIncrIter performs c += scalar ̅* vector, where vec and c both require iterators.
func MulSVIncrIter[T Num](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "MulVSIncrIter")
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
			retVal[j] += s * vec[i]
		}
	}
	return
}

// DivVV does c :=  a ̅/ b
func DivVV[T Num](a, b, c []T) {
	a = a[:]
	b = b[:len(a)]
	c = c[:len(a)]

	for i := range a {
		c[i] = a[i] / b[i]
	}
}

// DivVS does c := vec ̅/ scalar. The scalar value is broadcasted across the vector for the operation
func DivVS[T Num](vec []T, s T, retVal []T) {
	vec = vec[:]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = vec[i] / s
	}
}

// DivSV does c := scalar ̅/ vector. The scalar value is broadcasted across the vector for the operation.
func DivSV[T Num](s T, vec []T, retVal []T) {
	vec = vec[:]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = s / vec[i]
	}
}

// DivVVIncr does c += a ̅/ b
func DivVVIncr[T Num](a, b, incr []T) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]

	for i := range a {
		incr[i] += a[i] / b[i]
	}
}

// DivVVIter does c := a ̅/ b, where a, b, and c requires the use of an iterator.
func DivVVIter[T Num](a, b, c []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, c, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "DivVVIter")
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
			c[k] = a[i] / b[j]
		}
	}
	return

}

// DivVVIncrIter does c += a ̅/ b, where a, b, and c requires the use of an iterator.
func DivVVIncrIter[T Num](a, b, incr []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, incr, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "DivVVIncrIter")
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
			incr[k] += a[i] / b[j]
		}
	}
	return
}

// DivVSIncr performs c += vec ̅/ scalar. The scalar value is broadcasted across the vector for the operation.
func DivVSIncr[T Num](vec []T, s T, incr []T) {
	vec = vec[:]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += vec[i] / s
	}
}

// DivSVIncr performs c += scalar ̅/ vector. The scalar value is broadcasted across the vector for the operation.
func DivSVIncr[T Num](s T, vec []T, incr []T) {
	vec = vec[:]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += s / vec[i]
	}
}

// DivVSIter performs c := vec ̅/ scalar, where vec and c both require iterators.
func DivVSIter[T Num](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "DivVSIter")
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
			retVal[j] = vec[i] / s
		}
	}
	return
}

// DivSVIter performs c := scalar ̅/ vector, where vec and c both require iterators.
func DivSVIter[T Num](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "DivVSIter")
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
			retVal[j] = s / vec[i]
		}
	}
	return
}

// DivVSIncrIter performs c += vec ̅/ scalar, where vec and c both require iterators.
func DivVSIncrIter[T Num](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "DivVSIncrIter")
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
			retVal[j] += vec[i] / s
		}
	}
	return
}

// DivSVIncrIter performs c += scalar ̅/ vector, where vec and c both require iterators.
func DivSVIncrIter[T Num](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "DivVSIncrIter")
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
			retVal[j] += s / vec[i]
		}
	}
	return
}
