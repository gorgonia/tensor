package dense

import (
	"bytes"
	"encoding/gob"
	"io/ioutil"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
)

func TestDense_Gob(t *testing.T) {
	T := New[float32](WithShape(3, 2), WithBacking([]float32{0, 1, 2, 3, 4, 5}))

	var buf bytes.Buffer

	enc := gob.NewEncoder(&buf)
	err := enc.Encode(T)
	if err != nil {
		t.Fatal(err)
	}

	var U Dense[float32]
	dec := gob.NewDecoder(&buf)
	if err = dec.Decode(&U); err != nil {
		t.Fatal(err)
	}

	if !T.Eq(&U) {
		t.Errorf("Expected equal")
	}
}

func TestDense_Numpy(t *testing.T) {
	assert := assert.New(t)

	// BINARY COMPATIBILITY IS WRITTEN FOR NUMPY 1.21.5

	t.Run("(2,3) int array", func(t *testing.T) {
		f, err := os.Open("testdata/test_int.npy")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		T1, err := FromNpy(f)
		if err != nil {
			t.Fatal(err)
		}

		assert.True(shapes.Shape{2, 3}.Eq(T1.Shape()))
		assert.Equal([]int{1, 5, -1, 200, -10, 0}, T1.(*Dense[int]).Data())

		g, err := os.OpenFile("testdata/test_int_write.npy", os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0644)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Close()

		T1Int := T1.(*Dense[int])
		if err = T1Int.WriteNpy(g); err != nil {
			t.Fatal(err)
		}

		// Here we test that we write the correct npy file and that it is exactly binary compatible with what was written.
		//
		// The binary compatibility test removes the requirement for having Python in the test
		f.Seek(0, 0)
		g.Seek(0, 0)
		fbytes, err := ioutil.ReadAll(f)
		if err != nil {
			t.Fatal(err)
		}
		gbytes, err := ioutil.ReadAll(g)
		if err != nil {
			t.Fatal(err)
		}

		assert.Equal(fbytes, gbytes)
	})

	t.Run("(2,3) uint array", func(t *testing.T) {
		f, err := os.Open("testdata/test_uint64_(3, 2).npy")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		T1, err := FromNpy(f)
		if err != nil {
			t.Fatal(err)
		}

		assert.True(shapes.Shape{3, 2}.Eq(T1.Shape()))
		assert.Equal([]uint{0, 1, 2, 3, 4, 5}, T1.(*Dense[uint]).Data())

		g, err := os.OpenFile("testdata/test_uint_write.npy", os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0644)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Close()

		T1Uint := T1.(*Dense[uint])
		if err = T1Uint.WriteNpy(g); err != nil {
			t.Fatal(err)
		}

		// Here we test that we write the correct npy file and that it is exactly binary compatible with what was written.
		//
		// The binary compatibility test removes the requirement for having Python in the test
		f.Seek(0, 0)
		g.Seek(0, 0)
		fbytes, err := ioutil.ReadAll(f)
		if err != nil {
			t.Fatal(err)
		}
		gbytes, err := ioutil.ReadAll(g)
		if err != nil {
			t.Fatal(err)
		}

		assert.Equal(fbytes, gbytes)
	})

	t.Run("(2,3) float64 array", func(t *testing.T) {
		f, err := os.Open("testdata/test_float64_(2, 3).npy")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		T1, err := FromNpy(f)
		if err != nil {
			t.Fatal(err)
		}

		assert.True(shapes.Shape{2, 3}.Eq(T1.Shape()))
		assert.Equal([]float64{3.1415, 0, 12412245235, 2.717, -2345, -0.005}, T1.(*Dense[float64]).Data())

		g, err := os.OpenFile("testdata/test_float64_(2, 3)_write.npy", os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0644)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Close()

		T1Int := T1.(*Dense[float64])
		if err = T1Int.WriteNpy(g); err != nil {
			t.Fatal(err)
		}

		// Here we test that we write the correct npy file and that it is exactly binary compatible with what was written.
		//
		// The binary compatibility test removes the requirement for having Python in the test
		f.Seek(0, 0)
		g.Seek(0, 0)
		fbytes, err := ioutil.ReadAll(f)
		if err != nil {
			t.Fatal(err)
		}
		gbytes, err := ioutil.ReadAll(g)
		if err != nil {
			t.Fatal(err)
		}

		assert.Equal(fbytes, gbytes)
	})

	t.Run("(5,) int array", func(t *testing.T) {
		f, err := os.Open("testdata/test_int64_(5,).npy")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		T1, err := FromNpy(f)
		if err != nil {
			t.Fatal(err)
		}

		assert.True(shapes.Shape{5}.Eq(T1.Shape()))
		assert.Equal([]int{1, 2, 3, 4, 5}, T1.(*Dense[int]).Data())

		g, err := os.OpenFile("testdata/test_int_(5,)_write.npy", os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0644)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Close()

		T1Int := T1.(*Dense[int])
		if err = T1Int.WriteNpy(g); err != nil {
			t.Fatal(err)
		}

		// Here we test that we write the correct npy file and that it is exactly binary compatible with what was written.
		//
		// The binary compatibility test removes the requirement for having Python in the test
		f.Seek(0, 0)
		g.Seek(0, 0)
		fbytes, err := ioutil.ReadAll(f)
		if err != nil {
			t.Fatal(err)
		}
		gbytes, err := ioutil.ReadAll(g)
		if err != nil {
			t.Fatal(err)
		}

		assert.Equal(fbytes, gbytes)
	})
}
