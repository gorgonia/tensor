package tensor

import (
	"testing"
	"testing/quick"
)

func TestFloat32Engine_makeArray(t *testing.T) {

	// the uint16 is just to make sure that tests are correctly run.
	// we don't want the quicktest to randomly generate a size that is so large
	// that Go takes a long time just to allocate. We'll test the other sizes (like negative numbers)
	// after the quick test.
	f := func(sz uint16) bool {
		size := int(sz)
		e := Float32Engine{StdEng{}}
		dt := Float32
		arr := array{}

		e.makeArray(&arr, dt, size)

		if len(arr.Raw) != size*4 {
			t.Errorf("Expected raw to be size*4. Got %v instead", len(arr.Raw))
			return false
		}
		v, ok := arr.Data().([]float32)
		if !ok {
			t.Errorf("Expected v to be []float32. Got %T instead", arr.Data())
			return false
		}

		if len(v) != size {
			return false
		}
		return true
	}

	if err := quick.Check(f, nil); err != nil {
		t.Errorf("Quick test failed %v", err)
	}

}
