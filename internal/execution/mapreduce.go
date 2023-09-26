package execution

import "github.com/chewxy/inigo/values/tensor/internal/errors"

func MapIter[T any](fn func(T) T, a, retVal []T, ait, rit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsInput bool
	if retIsInput, err = iterCheck2(a, retVal, ait, rit); err != nil {
		return errors.Wrap(err, "MapIter")
	}
loop:
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}

		if retIsInput {
			j = i
			validj = validi
		} else {
			if j, validj, err = rit.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}
		if validi && validj {
			retVal[j] = fn(a[i])
		}

	}
	return
}

func MapIterWithErr[T any](fn func(T) (T, error), a, retVal []T, ait, rit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	var retIsInput bool
	if retIsInput, err = iterCheck2(a, retVal, ait, rit); err != nil {
		return errors.Wrap(err, "MapIter")
	}
loop:
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}

		if retIsInput {
			j = i
			validj = validi
		} else {
			if j, validj, err = rit.NextValidity(); err != nil {
				err = handleNoOp(err)
				break loop
			}
		}
		if validi && validj {
			if retVal[j], err = fn(a[i]); err != nil {
				return err
			}
		}

	}
	return
}

func Map[T any](fn func(T) T, a, retVal []T) (err error) {
	for i := range a {
		retVal[i] = fn(a[i])
	}
	return nil
}

func MapWithErr[T any](fn func(T) (T, error), a, retVal []T) (err error) {
	for i := range a {
		if retVal[i], err = fn(a[i]); err != nil {
			return err
		}
	}
	return nil
}
