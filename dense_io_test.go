package tensor

import (
	"bytes"
	"encoding/gob"
	"os"
	"os/exec"
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSaveLoadNumpy(t *testing.T) {
	if os.Getenv("CI_NO_PYTHON") == "true" {
		t.Skip("skipping test; This is being run on a CI tool that does not have Python")
	}

	assert := assert.New(t)
	T := New(WithShape(2, 2), WithBacking([]float64{1, 5, 10, -1}))
	// also checks the 1D Vector.
	T1D := New(WithShape(4), WithBacking([]float64{1, 5, 10, -1}))

	f, _ := os.OpenFile("test.npy", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	f1D, _ := os.OpenFile("test1D.npy", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)

	T.WriteNpy(f)
	f.Close()

	T1D.WriteNpy(f1D)
	f1D.Close()

	script := "import numpy as np\nx = np.load('test.npy')\nprint(x)\nx = np.load('test1D.npy')\nprint(x)"
	// Configurable python command, in order to be able to use python or python3
	pythonCommand := os.Getenv("PYTHON_COMMAND")
	if pythonCommand == "" {
		pythonCommand = "python"
	}

	cmd := exec.Command(pythonCommand)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Error(err)
	}

	go func() {
		defer stdin.Close()
		stdin.Write([]byte(script))
	}()

	buf := new(bytes.Buffer)
	cmd.Stdout = buf

	if err = cmd.Start(); err != nil {
		t.Error(err)
		t.Logf("Do you have a python with numpy installed? You can change the python interpreter by setting the environment variable PYTHON_COMMAND. Current value: PYTHON_COMMAND=%s", pythonCommand)
	}

	if err := cmd.Wait(); err != nil {
		t.Error(err)
	}

	expected := `\[\[\s*1\.\s*5\.\]\n \[\s*10\.\s*-1\.\]\]\n`
	if ok, _ := regexp.Match(expected, buf.Bytes()); !ok {
		t.Errorf("Did not successfully read numpy file, \n%q\n%q", buf.String(), expected)
	}

	if buf.String() != expected {
	}

	// cleanup
	err = os.Remove("test.npy")
	if err != nil {
		t.Error(err)
	}

	err = os.Remove("test1D.npy")
	if err != nil {
		t.Error(err)
	}

	// ok now to test if it can read
	T2 := new(Dense)
	buf = new(bytes.Buffer)
	T.WriteNpy(buf)
	if err = T2.ReadNpy(buf); err != nil {
		t.Fatal(err)
	}
	assert.Equal(T.Shape(), T2.Shape())
	assert.Equal(T.Strides(), T2.Strides())
	assert.Equal(T.Data(), T2.Data())

	// ok now to test if it can read 1D
	T1D2 := new(Dense)
	buf = new(bytes.Buffer)
	T1D.WriteNpy(buf)
	if err = T1D2.ReadNpy(buf); err != nil {
		t.Fatal(err)
	}
	assert.Equal(T1D.Shape(), T1D2.Shape())
	assert.Equal(T1D.Strides(), T1D2.Strides())
	assert.Equal(T1D.Data(), T1D2.Data())

	// try with masked array. masked elements should be filled with default value
	T.ResetMask(false)
	T.mask[0] = true
	T3 := new(Dense)
	buf = new(bytes.Buffer)
	T.WriteNpy(buf)
	if err = T3.ReadNpy(buf); err != nil {
		t.Fatal(err)
	}
	assert.Equal(T.Shape(), T3.Shape())
	assert.Equal(T.Strides(), T3.Strides())
	data := T.Float64s()
	data[0] = T.FillValue().(float64)
	assert.Equal(data, T3.Data())

	// try with 1D masked array. masked elements should be filled with default value
	T1D.ResetMask(false)
	T1D.mask[0] = true
	T1D3 := new(Dense)
	buf = new(bytes.Buffer)
	T1D.WriteNpy(buf)
	if err = T1D3.ReadNpy(buf); err != nil {
		t.Fatal(err)
	}
	assert.Equal(T1D.Shape(), T1D3.Shape())
	assert.Equal(T1D.Strides(), T1D3.Strides())
	data = T1D.Float64s()
	data[0] = T1D.FillValue().(float64)
	assert.Equal(data, T1D3.Data())
}

func TestSaveLoadCSV(t *testing.T) {
	assert := assert.New(t)
	for _, gtd := range serializationTestData {
		if _, ok := gtd.([]complex64); ok {
			continue
		}
		if _, ok := gtd.([]complex128); ok {
			continue
		}

		buf := new(bytes.Buffer)

		T := New(WithShape(2, 2), WithBacking(gtd))
		if err := T.WriteCSV(buf); err != nil {
			t.Error(err)
		}

		T2 := new(Dense)
		if err := T2.ReadCSV(buf, As(T.t)); err != nil {
			t.Error(err)
		}

		assert.Equal(T.Shape(), T2.Shape(), "Test: %v", gtd)
		assert.Equal(T.Data(), T2.Data())

	}

	T := New(WithShape(2, 2), WithBacking([]float64{1, 5, 10, -1}))
	f, _ := os.OpenFile("test.csv", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	T.WriteCSV(f)
	f.Close()

	// cleanup
	err := os.Remove("test.csv")
	if err != nil {
		t.Error(err)
	}

	// try with masked array. masked elements should be filled with default value
	T.ResetMask(false)
	T.mask[0] = true
	f, _ = os.OpenFile("test.csv", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	T.WriteCSV(f)
	f.Close()

	// cleanup again
	err = os.Remove("test.csv")
	if err != nil {
		t.Error(err)
	}

}

var serializationTestData = []interface{}{
	[]int{1, 5, 10, -1},
	[]int8{1, 5, 10, -1},
	[]int16{1, 5, 10, -1},
	[]int32{1, 5, 10, -1},
	[]int64{1, 5, 10, -1},
	[]uint{1, 5, 10, 255},
	[]uint8{1, 5, 10, 255},
	[]uint16{1, 5, 10, 255},
	[]uint32{1, 5, 10, 255},
	[]uint64{1, 5, 10, 255},
	[]float32{1, 5, 10, -1},
	[]float64{1, 5, 10, -1},
	[]complex64{1, 5, 10, -1},
	[]complex128{1, 5, 10, -1},
	[]string{"hello", "world", "hello", "世界"},
}

func TestDense_GobEncodeDecode(t *testing.T) {
	assert := assert.New(t)
	var err error
	for _, gtd := range serializationTestData {
		buf := new(bytes.Buffer)
		encoder := gob.NewEncoder(buf)
		decoder := gob.NewDecoder(buf)

		T := New(WithShape(2, 2), WithBacking(gtd))
		if err = encoder.Encode(T); err != nil {
			t.Errorf("Error while encoding %v: %v", gtd, err)
			continue
		}

		T2 := new(Dense)
		if err = decoder.Decode(T2); err != nil {
			t.Errorf("Error while decoding %v: %v", gtd, err)
			continue
		}

		assert.Equal(T.Shape(), T2.Shape())
		assert.Equal(T.Strides(), T2.Strides())
		assert.Equal(T.Data(), T2.Data())

		// try with masked array. masked elements should be filled with default value
		buf = new(bytes.Buffer)
		encoder = gob.NewEncoder(buf)
		decoder = gob.NewDecoder(buf)

		T.ResetMask(false)
		T.mask[0] = true
		assert.True(T.IsMasked())
		if err = encoder.Encode(T); err != nil {
			t.Errorf("Error while encoding %v: %v", gtd, err)
			continue
		}

		T3 := new(Dense)
		if err = decoder.Decode(T3); err != nil {
			t.Errorf("Error while decoding %v: %v", gtd, err)
			continue
		}

		assert.Equal(T.Shape(), T3.Shape())
		assert.Equal(T.Strides(), T3.Strides())
		assert.Equal(T.Data(), T3.Data())
		assert.Equal(T.mask, T3.mask)

	}
}

func TestDense_FBEncodeDecode(t *testing.T) {
	assert := assert.New(t)
	for _, gtd := range serializationTestData {
		T := New(WithShape(2, 2), WithBacking(gtd))
		buf, err := T.FBEncode()
		if err != nil {
			t.Errorf("UNPOSSIBLE!: %v", err)
			continue
		}

		T2 := new(Dense)
		if err = T2.FBDecode(buf); err != nil {
			t.Errorf("Error while decoding %v: %v", gtd, err)
			continue
		}

		assert.Equal(T.Shape(), T2.Shape())
		assert.Equal(T.Strides(), T2.Strides())
		assert.Equal(T.Data(), T2.Data())

		// TODO: MASKED ARRAY
	}
}

func TestDense_PBEncodeDecode(t *testing.T) {
	assert := assert.New(t)
	for _, gtd := range serializationTestData {
		T := New(WithShape(2, 2), WithBacking(gtd))
		buf, err := T.PBEncode()
		if err != nil {
			t.Errorf("UNPOSSIBLE!: %v", err)
			continue
		}

		T2 := new(Dense)
		if err = T2.PBDecode(buf); err != nil {
			t.Errorf("Error while decoding %v: %v", gtd, err)
			continue
		}

		assert.Equal(T.Shape(), T2.Shape())
		assert.Equal(T.Strides(), T2.Strides())
		assert.Equal(T.Data(), T2.Data())

		// TODO: MASKED ARRAY
	}
}
