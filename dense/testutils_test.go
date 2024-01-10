package dense

import (
	"math"
	"math/cmplx"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/chewxy/math32"
	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

func tolerance[DT float64 | float32](a, b, e DT) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	// note: b is correct (expected) value, a is actual value.
	// make error tolerance a fraction of b, not a.
	if b != 0 {
		e = e * b
		if e < 0 {
			e = -e
		}
	}
	return d < e
}
func closeenough[DT float64 | float32](a, b DT) bool {
	var e DT
	switch any(e).(type) {
	case float64:
		e = DT(1e-8)
	case float32:
		e = DT(1e-2)
	}
	return tolerance(a, b, e)
}

func closef[DT float64 | float32](a, b DT) bool {
	var e DT
	switch any(e).(type) {
	case float64:
		e = DT(1e-14)
	case float32:
		e = DT(1e-5) // the number gotten from the cfloat standard. Haskell's Linear package uses 1e-6 for floats
	}
	return tolerance(a, b, e)
}

func veryclose[DT float64 | float32](a, b DT) bool {
	var e DT
	switch any(e).(type) {
	case float64:
		e = DT(1e-16)
	case float32:
		e = DT(1e-6) // from wiki
	}
	return tolerance(a, b, e)
}

func alikef64(a, b float64) bool {
	switch {
	case math.IsNaN(a) && math.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(a) == math.Signbit(b)
	}
	return false
}

func alikef32(a, b float32) bool {
	switch {
	case math32.IsNaN(a) && math32.IsNaN(b):
		return true
	case a == b:
		return math32.Signbit(a) == math32.Signbit(b)
	}
	return false
}

// taken from math/cmplx testxo
func cTolerance(a, b complex128, e float64) bool {
	d := cmplx.Abs(a - b)
	if b != 0 {
		e = e * cmplx.Abs(b)
		if e < 0 {
			e = -e
		}
	}
	return d < e
}

func cClose(a, b complex128) bool              { return cTolerance(a, b, 1e-14) }
func cSoclose(a, b complex128, e float64) bool { return cTolerance(a, b, e) }
func cVeryclose(a, b complex128) bool          { return cTolerance(a, b, 4e-16) }
func cAlike(a, b complex128) bool {
	switch {
	case cmplx.IsNaN(a) && cmplx.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(real(a)) == math.Signbit(real(b)) && math.Signbit(imag(a)) == math.Signbit(imag(b))
	}
	return false
}

func allClose(a, b interface{}, approxFn ...interface{}) bool {
	switch at := a.(type) {
	case []float64:
		closeness := closef[float64]
		var ok bool
		if len(approxFn) > 0 {
			if closeness, ok = approxFn[0].(func(a, b float64) bool); !ok {
				closeness = closef[float64]
			}
		}
		bt := b.([]float64)
		for i, v := range at {
			if math.IsNaN(v) {
				if !math.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if math.IsInf(v, 0) {
				if !math.IsInf(bt[i], 0) {
					return false
				}
				continue
			}
			if !closeness(v, bt[i]) {
				return false
			}
		}
		return true
	case []float32:
		closeness := closef[float32]
		var ok bool
		if len(approxFn) > 0 {
			if closeness, ok = approxFn[0].(func(a, b float32) bool); !ok {
				closeness = closef[float32]
			}
		}
		bt := b.([]float32)
		for i, v := range at {
			if math32.IsNaN(v) {
				if !math32.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if math32.IsInf(v, 0) {
				if !math32.IsInf(bt[i], 0) {
					return false
				}
				continue
			}
			if !closeness(v, bt[i]) {
				return false
			}
		}
		return true
	case []complex64:
		bt := b.([]complex64)
		for i, v := range at {
			if cmplx.IsNaN(complex128(v)) {
				if !cmplx.IsNaN(complex128(bt[i])) {
					return false
				}
				continue
			}
			if cmplx.IsInf(complex128(v)) {
				if !cmplx.IsInf(complex128(bt[i])) {
					return false
				}
				continue
			}
			if !cSoclose(complex128(v), complex128(bt[i]), 1e-5) {
				return false
			}
		}
		return true
	case []complex128:
		bt := b.([]complex128)
		for i, v := range at {
			if cmplx.IsNaN(v) {
				if !cmplx.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if cmplx.IsInf(v) {
				if !cmplx.IsInf(bt[i]) {
					return false
				}
				continue
			}
			if !cClose(v, bt[i]) {
				return false
			}
		}
		return true
	default:
		return reflect.DeepEqual(a, b)
	}
}

const quickchecks = 1000

func newRand() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

// qcDense generates a valid *Dense[DT].
func qcDense[DT internal.OrderedNum](args []reflect.Value, rnd *rand.Rand) {
	// get random shape
	var shape shapes.Shape
	for i := 0; i < rnd.Intn(4)+1; i++ {
		shape = append(shape, rnd.Intn(5)+1)
	}
	backing := make([]DT, shape.TotalSize())
	for i := range backing {
		backing[i] = DT(-1000 + rnd.Float64()*2000)
	}

	d := New[DT](WithShape(shape...), WithBacking(backing))

	// funny engines/memory access patterns
	x := rnd.Intn(10)
	switch x {
	case 1:
		// colmajor
		d.AP.SetDataOrder(tensor.ColMajor)
	case 2:
		// not accessible
		d.f |= tensor.NativelyInaccessible
	case 3:
	// transposed
	case 4:
	// requires iterator
	default:
		// nothing
	}
	args[0] = reflect.ValueOf(d)

	switch len(args) {
	case 2:
		// second argument is for broadcasting
		shp2 := shape.Clone()

		// < 50% chance of getting a shape that is bigger
		x := rnd.Intn(10)
		switch x {
		case 1, 2:
			shp2 = append(shp2, 0)
			copy(shp2[1:], shp2[0:])
			shp2[0] = 1
		case 3, 4:
			shp2 = append(shp2, 1)
		case 5, 6:
			shp2 = shp2[1:]
		default:
			for i := range shp2 {
				x := rnd.Intn(10)
				if x <= 3 {
					shp2[i] = 1
				}
			}
		}

		b := New[DT](WithShape(shp2...))
		args[1] = reflect.ValueOf(b)
	case 3:
		// 3 is reserved for testing scalar methods.
		s := DT(-1000 + rnd.Float64()*2000)
		args[1] = reflect.ValueOf(s)
		var scalarOnLeft bool
		if rnd.Intn(2) == 1 {
			scalarOnLeft = true
		}
		args[2] = reflect.ValueOf(scalarOnLeft)

	}

}

func qcErrCheck(t *testing.T, name string, a, b any, we bool, err error) (e error, retEarly bool) {
	switch {
	case !we && err != nil:
		t.Errorf("Tests for %v (%T) was unable to proceed: %v", name, a, err)
		return err, true
	case we && err == nil:
		if b == nil {
			t.Errorf("Expected error when performing %v on %T ", name, a)
			return errors.New("Error"), true
		}
		t.Errorf("Expected error when performing %v on %T  and %T", name, a, b)
		return errors.New("Error"), true
	case we && err != nil:
		return nil, true
	}
	return nil, false
}

func qcIsFloat(dt dtype.Dtype) bool {
	if err := dtype.TypeClassCheck(dt, dtype.FloatComplex); err == nil {
		return true
	}
	return false
}

func qcEqCheck(t *testing.T, dt dtype.Dtype, willFailEq bool, correct, got interface{}) bool {
	isFloatTypes := qcIsFloat(dt)
	if !willFailEq && (isFloatTypes && !allClose(correct, got) || (!isFloatTypes && !reflect.DeepEqual(correct, got))) {
		t.Errorf("q.Dtype: %v", dt)
		t.Errorf("correct\n%v", correct)
		t.Errorf("got\n%v", got)
		return false
	}
	return true
}

func checkErr(t *testing.T, expected bool, err error, name string, id interface{}) (cont bool) {
	switch {
	case expected:
		if err == nil {
			t.Errorf("Expected error in test %v (%v)", name, id)
		}
		return true
	case !expected && err != nil:
		t.Errorf("Test %v (%v) errored: %+v", name, id, err)
		return true
	}
	return false
}
