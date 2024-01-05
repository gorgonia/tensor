package main

import "text/template"

const executionArithRaw = `// {{.Name}}VV does c :=  a ̅{{.Symbol}} b
func {{.Name}}VV[T {{.TypeClass}}](a, b, c []T) {
	a = a[:len(a)]
	b = b[:len(a)]
	c = c[:len(a)]

	for i := range a {
		c[i] = a[i] {{.Symbol}} b[i]
	}
}

// {{.Name}}BC does :=  a ̅{{.Symbol}} b, using the appropriate indexing that follows a broadcast operation.
func {{.Name}}BC[T {{.TypeClass}}](a, b, c []T, aShp, bShp, cShp shapes.Shape, aStrides, bStrides []int) {
	for i := range c {
		var idxA, idxB int
		for j := range cShp {
			aDim, bDim := 1, 1
			if j < aShp.Dims() {
				aDim = aShp[j]
			}
			if j < bShp.Dims() {
				bDim = bShp[j]
			}
			idxDim := (i / cShp[j+1:].TotalSize()) % cShp[j]
			if aDim != 1 {
				idxA += (idxDim % aDim) * aStrides[j]
			}
			if bDim != 1 {
				idxB += (idxDim % bDim) * bStrides[j]
			}
		}
		c[i] = a[idxA] {{.Symbol}} b[idxB]
	}
}

// {{.Name}}VS does c := vec ̅{{.Symbol}} scalar. The scalar value is broadcasted across the vector for the operation
func {{.Name}}VS[T {{.TypeClass}}](vec []T, s T, retVal []T) {
	vec = vec[:len(vec)]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = vec[i] {{.Symbol}} s
	}
}

// {{.Name}}SV does c := scalar ̅{{.Symbol}} vector. The scalar value is broadcasted across the vector for the operation.
func {{.Name}}SV[T {{.TypeClass}}]( s T, vec []T, retVal []T) {
	vec = vec[:len(vec)]
	retVal = retVal[:len(vec)]
	for i := range vec {
		retVal[i] = s {{.Symbol}} vec[i]
	}
}

// {{.Name}}VVIncr does c += a ̅{{.Symbol}} b.
func {{.Name}}VVIncr[T {{.TypeClass}}](a, b, incr []T) {
	a = a[:len(a)]
	b = b[:len(a)]
	incr = incr[:len(a)]

	for i := range a {
		incr[i] += a[i] {{.Symbol}} b[i]
	}
}

// {{.Name}}BCIncr does c += a ̅{{.Symbol}} b, except it's broadcasted.
func {{.Name}}BCIncr[T {{.TypeClass}}](a, b, c []T, aShp, bShp, cShp shapes.Shape, aStrides, bStrides []int) {
	for i := range c {
		var idxA, idxB int
		for j := range cShp {
			aDim, bDim := 1, 1
			if j < aShp.Dims() {
				aDim = aShp[j]
			}
			if j < bShp.Dims() {
				bDim = bShp[j]
			}
			idxDim := (i / cShp[j+1:].TotalSize()) % cShp[j]
			if aDim != 1 {
				idxA += (idxDim % aDim) * aStrides[j]
			}
			if bDim != 1 {
				idxB += (idxDim % bDim) * bStrides[j]
			}
		}
		c[i] += a[idxA] {{.Symbol}} b[idxB]
	}
}


// {{.Name}}VVIter does c := a ̅{{.Symbol}} b, where a, b, and c requires the use of an iterator.
func {{.Name}}VVIter[T {{.TypeClass}}](a, b, c []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, c, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "{{.Name}}VVIter")
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
			c[k] = a[i] {{.Symbol}} b[j]
		}
	}
	return

}

// {{.Name}}VVIncrIter does c += a ̅{{.Symbol}} b, where a, b, and c requires the use of an iterator.
func {{.Name}}VVIncrIter[T {{.TypeClass}}](a, b, incr []T, ait, bit, cit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	var cisa, cisb bool
	if cisa, cisb, err = iterCheck3(a, b, incr, ait, bit, cit); err != nil {
		return errors.Wrapf(err, "{{.Name}}VVIncrIter")
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
			incr[k] += a[i] {{.Symbol}} b[j]
		}
	}
	return
}

// {{.Name}}VSIncr performs c += vec ̅{{.Symbol}} scalar. The scalar value is broadcasted across the vector for the operation.
func {{.Name}}VSIncr[T {{.TypeClass}}](vec []T, s T, incr []T) {
	vec = vec[:len(vec)]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += vec[i] {{.Symbol}} s
	}
}

// {{.Name}}SVIncr performs c += scalar ̅{{.Symbol}} vector. The scalar value is broadcasted across the vector for the operation.
func {{.Name}}SVIncr[T {{.TypeClass}}](s T, vec []T, incr []T) {
	vec = vec[:len(vec)]
	incr = incr[:len(vec)]
	for i := range vec {
		incr[i] += s {{.Symbol}} vec[i]
	}
}

// {{.Name}}VSIter performs c := vec ̅{{.Symbol}} scalar, where vec and c both require iterators.
func {{.Name}}VSIter[T {{.TypeClass}}](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "{{.Name}}VSIter")
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
			retVal[j] = vec[i] {{.Symbol}} s
		}
	}
	return
}


// {{.Name}}SVIter performs c := scalar ̅{{.Symbol}} vector, where vec and c both require iterators.
func {{.Name}}SVIter[T {{.TypeClass}}](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "{{.Name}}VSIter")
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
			retVal[j] = s {{.Symbol}} vec[i]
		}
	}
	return
}

// {{.Name}}VSIncrIter performs c += vec ̅{{.Symbol}} scalar, where vec and c both require iterators.
func {{.Name}}VSIncrIter[T {{.TypeClass}}](vec []T, s T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "{{.Name}}VSIncrIter")
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
			retVal[j] += vec[i] {{.Symbol}} s
		}
	}
	return
}

// {{.Name}}SVIncrIter performs c += scalar ̅{{.Symbol}} vector, where vec and c both require iterators.
func {{.Name}}SVIncrIter[T {{.TypeClass}}](s T, vec []T, retVal []T, veciter, retiter Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(vec, retVal, veciter, retiter); err != nil {
		return errors.Wrapf(err, "{{.Name}}VSIncrIter")
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
			retVal[j] += s {{.Symbol}} vec[i]
		}
	}
	return
}


`

const executionCmpRaw = `// {{.Name}}VV performs c := a ̅{{.Symbol}} b where c is of the same type as the inputs.
func {{.Name}}VV[T {{.TypeClass}}](a, b, c []T) {
	a = a[:len(a)]
	b = b[:len(a)]
	c = c[:len(a)]

	for i := range a {
		if a[i] {{.Symbol}} b[i] {
			c[i] = T(1)
		} else {
			c[i] = T(0)
		}
	}
}

// {{.Name}}BC performs c := a ̅{{.Symbol}} b where c is of the same type as the inputs, using the appropriate indexing that follows a broadcast operation.
func {{.Name}}BC[T {{.TypeClass}}](a, b, c []T, aShp, bShp, cShp shapes.Shape, aStrides, bStrides []int) {
	for i := range c {
		var idxA, idxB int
		for j := range cShp {
			aDim, bDim := 1, 1
			if j < aShp.Dims() {
				aDim = aShp[j]
			}
			if j < bShp.Dims() {
				bDim = bShp[j]
			}
			idxDim := (i / cShp[j+1:].TotalSize()) % cShp[j]
			if aDim != 1 {
				idxA += (idxDim % aDim) * aStrides[j]
			}
			if bDim != 1 {
				idxB += (idxDim % bDim) * bStrides[j]
			}
		}
		if a[i] {{.Symbol}} b[i] {
			c[i] = T(1)
		} else {
			c[i] = T(0)
		}
	}
}

// {{.Name}}VVIter performs c := a  ̅{{.Symbol}} b, where a, b, and c requires the use of an iterator.
func {{.Name}}VVIter[T {{.TypeClass}}](a, b, c []T, ait, bit, cit Iterator) (err error) {
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
			if a[i] {{.Symbol}} b[j] {
				c[k] = T(1)
			} else {
				c[k] = T(0)
			}
		}
	}
	return
}

// {{.Name}}VS performs c := vec ̅{{.Symbol}} scalar where c is of the same datatype as its inputs. The scalar value is broadcasted for the operation.
func {{.Name}}VS[T {{.TypeClass}}](a []T, b T, c []T) {
	a = a[:len(a)]
	c = c[:len(a)]

	for i := range a {
		if a[i] {{.Symbol}} b {
			c[i] = T(1)
		} else {
			c[i] = T(0)
		}
	}
}


// {{.Name}}VSIter performs c := vec ̅{{.Symbol}} scalar where c is of the same datatype as the inputs. The scalar value is broadcasted for the operation.
func {{.Name}}VSIter[T {{.TypeClass}}](a []T, b T, c []T, ait, cit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(a, c, ait, cit); err != nil {
		return errors.Wrapf(err, internal.ThisFn())
	}
loop:
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if retIsVec {
			j = i
			validj = validi
		} else {
			if j, validj, err = cit.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj {
			if a[i] {{.Symbol}} b {
				c[j] = T(1)
			} else {
				c[j] = T(0)
			}
		}
	}
	return

}

// {{.Name}}SV performs c := scalar ̅{{.Symbol}} vec where c is of the same datatype as the inputs. The scalar value is broadcasted for the operation.
func {{.Name}}SV[T {{.TypeClass}}](a T, b []T, c []T) {
	b = b[:len(b)]
	c = c[:len(b)]

	for i := range b {
		if a {{.Symbol}} b[i] {
			c[i] = T(1)
		} else {
			c[i] = T(0)
		}
	}
}

// {{.Name}}SVIter performs c := scalar ̅{{.Symbol}} vec where c is of the same datatype as the inputs. The scalar value is broadcasted for the operation.
func {{.Name}}SVIter[T {{.TypeClass}}](a T, b []T, C []T, bit, cit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(b, C, bit, cit); err != nil {
		return errors.Wrapf(err, internal.ThisFn())
	}
loop:
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if retIsVec {
			j = i
			validj = validi
		} else {
			if j, validj, err = cit.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj {
			if a {{.Symbol}} b[i] {
				C[j] = T(1)
			} else {
				C[j] = T(0)
			}
		}
	}
	return

}
`

const executionCmpBoolRaw = `// {{.Name}}VVBool performs c := a ̅{{.Symbol}} b.
func {{.Name}}VVBool[T {{.TypeClass}}](a, b []T, c []bool) {
	a = a[:len(a)]
	b = b[:len(a)]
	c = c[:len(a)]

	for i := range a {
		c[i] = a[i] {{.Symbol}} b[i]
	}
}

// {{.Name}}BCBool performs c := a ̅{{.Symbol}} b, using the appropriate indexing that follows a broadcast operation.
func {{.Name}}BCBool[T {{.TypeClass}}](a, b []T, c []bool, aShp, bShp, cShp shapes.Shape, aStrides, bStrides []int) {
	for i := range c {
		var idxA, idxB int
		for j := range cShp {
			aDim, bDim := 1, 1
			if j < aShp.Dims() {
				aDim = aShp[j]
			}
			if j < bShp.Dims() {
				bDim = bShp[j]
			}
			idxDim := (i / cShp[j+1:].TotalSize()) % cShp[j]
			if aDim != 1 {
				idxA += (idxDim % aDim) * aStrides[j]
			}
			if bDim != 1 {
				idxB += (idxDim % bDim) * bStrides[j]
			}
		}
		c[i] = a[idxA] {{.Symbol}} b[idxB]
	}
}

// {{.Name}}VVIterBool performs c := a  ̅{{.Symbol}} b, where a, b, and c requires the use of an iterator.
func {{.Name}}VVIterBool[T {{.TypeClass}}](a, b []T, c []bool, ait, bit, cit Iterator) (err error) {
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
			c[k] = a[i] {{.Symbol}} b[j]
		}
	}
	return
}

// {{.Name}}VSBool performs c := vec ̅{{.Symbol}} scalar. The scalar value is broadcasted for the operation.
func {{.Name}}VSBool[T {{.TypeClass}}](a []T, b T, c []bool) {
	a = a[:len(a)]
	c = c[:len(a)]

	for i := range a {
		c[i] = a[i] {{.Symbol}} b
	}
}

// {{.Name}}VSIterBool performs c := vec ̅{{.Symbol}} scalar. The scalar value is broadcasted for the operation.
func {{.Name}}VSIterBool[T {{.TypeClass}}](a []T, b T, c []bool, ait, cit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(a, c, ait, cit); err != nil {
		return errors.Wrapf(err, internal.ThisFn())
	}
loop:
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if retIsVec {
			j = i
			validj = validi
		} else {
			if j, validj, err = cit.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj {
			c[j] = a[i] {{.Symbol}} b
		}
	}
	return

}

// {{.Name}}SVBool performs c := scalar ̅{{.Symbol}} vec. The scalar value is broadcasted for the operation.
func {{.Name}}SVBool[T {{.TypeClass}}](a T, b []T, c []bool) {
	b = b[:len(b)]
	c = c[:len(b)]

	for i := range b {
		c[i] = a {{.Symbol}} b[i]
	}
}

// {{.Name}}SVIterBool performs c := scalar ̅{{.Symbol}} vec. The scalar value is broadcasted for the operation.
func {{.Name}}SVIterBool[T {{.TypeClass}}](a T, b []T, c []bool, bit, cit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsVec bool
	if retIsVec, err = iterCheck2(b, c, bit, cit); err != nil {
		return errors.Wrapf(err, internal.ThisFn())
	}
loop:
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if retIsVec {
			j = i
			validj = validi
		} else {
			if j, validj, err = cit.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}

		if validi && validj {
			c[j] = a {{.Symbol}} b[i]
		}
	}
	return

}
`

var (
	executionArith   *template.Template
	executionCmp     *template.Template
	executionCmpBool *template.Template
)

func init() {
	executionArith = template.Must(template.New("executionArith").Parse(executionArithRaw))
	executionCmp = template.Must(template.New("executionCmp").Parse(executionCmpRaw))
	executionCmpBool = template.Must(template.New("executionCmpBool").Parse(executionCmpBoolRaw))
}
