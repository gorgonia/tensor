package stdeng_test

import (
	"context"
	"fmt"
	"testing"

	"gorgonia.org/tensor/dense"
	. "gorgonia.org/tensor/engines"
	gutils "gorgonia.org/tensor/internal/utils"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
)

func ExampleStdEng_SelectByIndices() {
	d := dense.New[float64](WithShape(5, 3), WithBacking(gutils.Range[float64](1, 16)))
	indices := dense.New[int](WithShape(8), WithBacking([]int{1, 1, 4, 4, 0, 0, 0, 0}))
	retVal := dense.New[float64](WithShape(8, 3))
	e := StdEng[float64, *dense.Dense[float64]]{}

	fmt.Printf("Let d =\n%v\n", d)

	// Case 0: outermost selection (the most common use case)
	fmt.Println("The most common usecase of `SelectByIndices`: Outermost selection")
	if err := e.SelectByIndices(context.Background(), d, indices, 0, retVal); err != nil {
		fmt.Printf("Error: %v", err)
		return
	}
	fmt.Printf("Selecting indices %v from axis %d of d yields:\n%v\n-----\n", indices, 0, retVal)

	// Case 1: innermost selection
	fmt.Println("Innermost selection, note the difference in behaviour")
	indices = dense.New[int](WithShape(8), WithBacking([]int{1, 1, 2, 2, 0, 0, 0, 0}))
	retVal = dense.New[float64](WithShape(5, 8))
	if err := e.SelectByIndices(context.Background(), d, indices, 1, retVal); err != nil {
		fmt.Printf("Error: %v", err)
		return
	}
	fmt.Printf("Selecting indices %v from axis %d of d yields:\n%v.", indices, 1, retVal)

	// Output:
	// Let d =
	// ⎡ 1   2   3⎤
	// ⎢ 4   5   6⎥
	// ⎢ 7   8   9⎥
	// ⎢10  11  12⎥
	// ⎣13  14  15⎦
	//
	// The most common usecase of `SelectByIndices`: Outermost selection
	// Selecting indices [1  1  4  4  0  0  0  0] from axis 0 of d yields:
	// ⎡ 4   5   6⎤
	// ⎢ 4   5   6⎥
	// ⎢13  14  15⎥
	// ⎢13  14  15⎥
	// ⎢ 1   2   3⎥
	// ⎢ 1   2   3⎥
	// ⎢ 1   2   3⎥
	// ⎣ 1   2   3⎦
	//
	// -----
	// Innermost selection, note the difference in behaviour
	// Selecting indices [1  1  2  2  0  0  0  0] from axis 1 of d yields:
	// ⎡ 2   2   3   3   1   1   1   1⎤
	// ⎢ 5   5   6   6   4   4   4   4⎥
	// ⎢ 8   8   9   9   7   7   7   7⎥
	// ⎢11  11  12  12  10  10  10  10⎥
	// ⎣14  14  15  15  13  13  13  13⎦
	// .
}

func ExampleStdNumEngine_SelectByIndicesB() {
	d := dense.New[float64](WithShape(5, 3), WithBacking(gutils.Range[float64](1, 16)))
	indices := dense.New[int](WithShape(8), WithBacking([]int{1, 1, 4, 4, 0, 0, 0, 0}))
	retVal := dense.New[float64](WithShape(8, 3))
	e := StdNumEngine[float64, *dense.Dense[float64]]{}

	fmt.Printf("Let d =\n%v\n", d)

	// Case 0: outermost selection (the most common use case)
	fmt.Println("The most common usecase of `SelectByIndices`: Outermost selection")
	if err := e.SelectByIndices(context.Background(), d, indices, 0, retVal); err != nil {
		fmt.Printf("Error: %v", err)
		return
	}
	fmt.Printf("Selecting indices %v from axis %d of d yields:\n%v\n-----\n", indices, 0, retVal)

	outGrad := retVal.Clone()
	if err := outGrad.Memset(1); err != nil {
		fmt.Println(err)
		return
	}
	retGrad := d.Clone()
	retGrad.Zero()
	if err := e.SelectByIndicesB(context.Background(), d, outGrad, indices, 0, retGrad); err != nil {
		fmt.Printf("Error %v", err)
		return
	}
	fmt.Printf("Grad\n%v", retGrad)

	// Output:
	// Let d =
	// ⎡ 1   2   3⎤
	// ⎢ 4   5   6⎥
	// ⎢ 7   8   9⎥
	// ⎢10  11  12⎥
	// ⎣13  14  15⎦
	//
	// The most common usecase of `SelectByIndices`: Outermost selection
	// Selecting indices [1  1  4  4  0  0  0  0] from axis 0 of d yields:
	// ⎡ 4   5   6⎤
	// ⎢ 4   5   6⎥
	// ⎢13  14  15⎥
	// ⎢13  14  15⎥
	// ⎢ 1   2   3⎥
	// ⎢ 1   2   3⎥
	// ⎢ 1   2   3⎥
	// ⎣ 1   2   3⎦
	//
	// -----
	// Grad
	// ⎡4  4  4⎤
	// ⎢2  2  2⎥
	// ⎢0  0  0⎥
	// ⎢0  0  0⎥
	// ⎣2  2  2⎦
}

type selectByIndicesTest struct {
	Name    string
	Data    []float64
	Shape   shapes.Shape
	Indices []int
	Axis    int

	WillErr      bool
	Correct      []float64
	CorrectShape shapes.Shape
}

var selectByIndicesTests = []selectByIndicesTest{
	{Name: "Basic", Data: gutils.Range[float64](0, 4), Shape: shapes.Shape{2, 2}, Indices: []int{0, 1}, Axis: 0, WillErr: false,
		Correct: []float64{0, 1, 2, 3}, CorrectShape: shapes.Shape{2, 2}},
	{Name: "3-tensor, axis 0", Data: gutils.Range[float64](0, 24), Shape: shapes.Shape{3, 2, 4}, Indices: []int{1, 1}, Axis: 0, WillErr: false,
		Correct: []float64{8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15}, CorrectShape: shapes.Shape{2, 2, 4}},

	{Name: "3-tensor, axis 1", Data: gutils.Range[float64](0, 24), Shape: shapes.Shape{3, 2, 4}, Indices: []int{1, 1}, Axis: 1, WillErr: false,
		Correct: []float64{4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15, 20, 21, 22, 23, 20, 21, 22, 23}, CorrectShape: shapes.Shape{3, 2, 4}},

	{Name: "3-tensor, axis 2", Data: gutils.Range[float64](0, 24), Shape: shapes.Shape{3, 2, 4}, Indices: []int{1, 1}, Axis: 2, WillErr: false,
		Correct: []float64{1, 1, 5, 5, 9, 9, 13, 13, 17, 17, 21, 21}, CorrectShape: shapes.Shape{3, 2, 2}},

	{Name: "Vector, axis 0", Data: gutils.Range[float64](0, 5), Shape: shapes.Shape{5}, Indices: []int{1, 1}, Axis: 0, WillErr: false,
		Correct: []float64{1, 1}, CorrectShape: shapes.Shape{2}},

	{Name: "Vector, axis 1", Data: gutils.Range[float64](0, 5), Shape: shapes.Shape{5}, Indices: []int{1, 1}, Axis: 1, WillErr: true,
		Correct: []float64{1, 1}, CorrectShape: shapes.Shape{2}},
	{Name: "(4,2) Matrix, with (10) indices", Data: gutils.Range[float64](0, 8), Shape: shapes.Shape{4, 2}, Indices: []int{1, 1, 1, 1, 0, 2, 2, 2, 2, 0}, Axis: 0, WillErr: false,
		Correct: []float64{2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 4, 5, 4, 5, 4, 5, 4, 5, 0, 1}, CorrectShape: shapes.Shape{10, 2}},
	{Name: "(2,1) Matrx (colvec) with (10) indices", Data: gutils.Range[float64](0, 2), Shape: shapes.Shape{2, 1}, Indices: []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, Axis: 0, WillErr: false,
		Correct: []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, CorrectShape: shapes.Shape{10},
	},
}

func TestStdEng_SelectByIndices(t *testing.T) {
	assert := assert.New(t)
	for i, tc := range selectByIndicesTests {
		d := dense.New[float64](WithShape(tc.Shape...), WithBacking(tc.Data))
		indices := dense.New[int](WithBacking(tc.Indices))
		retVal := dense.New[float64](WithShape(tc.CorrectShape...))
		e := StdEng[float64, *dense.Dense[float64]]{}
		err := e.SelectByIndices(context.Background(), d, indices, tc.Axis, retVal)
		switch {
		case tc.WillErr && err == nil:
			t.Errorf("Expected error in %v(%d)", tc.Name, i)
			continue
		case tc.WillErr && err != nil:
			continue
		case !tc.WillErr && err != nil:
			t.Errorf("Err in %v(%d): %v", tc.Name, i, err)
			continue
		default:
		}

		assert.Equal(tc.Correct, retVal.Data())
		assert.True(tc.CorrectShape.Eq(retVal.Shape()))
	}
}

type selectByIndicesBTest struct {
	selectByIndicesTest

	CorrectGrad      []float64
	CorrectGradShape shapes.Shape
}

var selectByIndicesBTests = []selectByIndicesBTest{
	{CorrectGrad: []float64{1, 1, 1, 1}},
	// 3-tensor, axis 0
	{CorrectGrad: []float64{0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0}},
	// 3-tensor, axis 1
	{CorrectGrad: []float64{0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2}},
	// 3-tensor, axis 2
	{CorrectGrad: []float64{0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0}},
	// vector, axis 0
	{CorrectGrad: []float64{0, 2, 0, 0, 0}},
	// vector, axis 1
	{CorrectGrad: nil}, // []float64{4, 6, 8, 12, 8, 12, 0, 0} if it didn't error
	// (4,2) Matrix with (10) indices
	{CorrectGrad: []float64{2, 2, 4, 4, 4, 4, 0, 0}},
	// (2, 1) Matrix (colvec) with (10) indices
	{CorrectGrad: []float64{0, 10}},
}

func init() {
	for i := range selectByIndicesBTests {
		selectByIndicesBTests[i].selectByIndicesTest = selectByIndicesTests[i]
		selectByIndicesBTests[i].CorrectGradShape = selectByIndicesTests[i].Shape
	}
}

func TestStdEng_SelectByIndicesB(t *testing.T) {
	assert := assert.New(t)

	for i, tc := range selectByIndicesBTests {
		d := dense.New[float64](WithShape(tc.Shape...), WithBacking(tc.Data))
		indices := dense.New[int](WithBacking(tc.Indices))
		retVal := dense.New[float64](WithShape(tc.CorrectShape...))
		e := StdEng[float64, *dense.Dense[float64]]{}
		err := e.SelectByIndices(context.Background(), d, indices, tc.Axis, retVal)
		switch {
		case tc.WillErr && err == nil:
			t.Errorf("Expected error in %v(%d)", tc.Name, i)
			continue
		case tc.WillErr && err != nil:
			continue
		case !tc.WillErr && err != nil:
			t.Errorf("Err in %v(%d): %v", tc.Name, i, err)
			continue
		default:
		}

		outGrad := retVal.Clone()
		if err = outGrad.Memset(1.0); err != nil {
			t.Errorf("err while memsetting in %v: %v", tc.Name, err)
			continue
		}
		retGrad := dense.New[float64](WithShape(tc.CorrectGradShape...))

		e2 := StdNumEngine[float64, *dense.Dense[float64]]{}
		err = e2.SelectByIndicesB(context.Background(), d, outGrad, indices, tc.Axis, retGrad)
		if err != nil {
			t.Errorf("error in computing backward pass %v: %v", tc.Name, err)
			continue
		}
		assert.Equal(tc.CorrectGrad, retGrad.Data(), "Test %v", tc.Name)
	}
}
