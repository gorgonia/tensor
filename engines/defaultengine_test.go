package stdeng_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
	. "gorgonia.org/tensor/engines"
	"gorgonia.org/tensor/internal"
	gutils "gorgonia.org/tensor/internal/utils"
)

func TestStdEng_Reduce(t *testing.T) {
	assert := assert.New(t)

	a := dense.New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	eng := a.Engine().(tensor.Reducer[float64, *dense.Dense[float64]])

	ctx := context.Background()

	module := tensor.ReductionModule[float64]{
		ReduceFirstN: func(retVal, a []float64) {
			for i := range retVal {
				retVal[i] += a[i]
			}
		},
		ReduceLastN: func(a []float64, defaultValue float64) float64 {
			retVal := defaultValue
			for i := range a {
				retVal += a[i]
			}
			return retVal
		},
		Reduce: func(a, b float64) float64 { return a + b },
	}
	plusWithErr := func(a, b float64) (float64, error) { return a + b, nil }

	/* Reduce First Axis */

	correctSum0 := []float64{5, 7, 9}

	// Sum with ReduceFirstN
	retVal := dense.New[float64](WithShape(3))
	err := eng.Reduce(ctx, module.ReduceFirstN, a, 0, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum0, retVal.Data())

	// Sum with ReduceFirst
	retVal = dense.New[float64](WithShape(3))
	err = eng.Reduce(ctx, module.Reduce, a, 0, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum0, retVal.Data())

	// Sum with ReduceFirstWithErr
	retVal = dense.New[float64](WithShape(3))
	err = eng.Reduce(ctx, plusWithErr, a, 0, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum0, retVal.Data())

	// Sum With ReductionFunctions
	retVal = dense.New[float64](WithShape(3))
	err = eng.Reduce(ctx, module, a, 0, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum0, retVal.Data())

	/* Reduce Last Axis */

	correctSum1 := []float64{6, 15}
	// Sum With ReduceLastN
	retVal = dense.New[float64](WithShape(2))
	err = eng.Reduce(ctx, module.ReduceLastN, a, 1, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum1, retVal.Data())

	// Sum With ReduceLast
	retVal = dense.New[float64](WithShape(2))
	err = eng.Reduce(ctx, module.Reduce, a, 1, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum1, retVal.Data())

	// Sum With ReduceLastWithErr
	retVal = dense.New[float64](WithShape(2))
	err = eng.Reduce(ctx, plusWithErr, a, 1, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum1, retVal.Data())

	// Sum With ReductionFunctions
	retVal = dense.New[float64](WithShape(2))
	err = eng.Reduce(ctx, module, a, 1, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum1, retVal.Data())

	/* Reduction on Arbitrary Axis */

	// 3D tensor in order to test arbitrary axis reduction
	a = dense.New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))

	correctSum1 = []float64{12, 15, 18, 21, 48, 51, 54, 57}

	// Sum over axis 1 using plus
	retVal = dense.New[float64](WithShape(2, 4))
	err = eng.Reduce(ctx, module.Reduce, a, 1, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum1, retVal.Data())

	// Sum over axis 1 using plusWithErr
	retVal = dense.New[float64](WithShape(2, 4))
	err = eng.Reduce(ctx, plusWithErr, a, 1, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum1, retVal.Data())

	// Sum over axis 1 using ReductionFunctions
	retVal = dense.New[float64](WithShape(2, 4))
	err = eng.Reduce(ctx, module, a, 1, 0.0, retVal)
	assert.NoError(err)
	assert.Equal(correctSum1, retVal.Data())
}

func TestStdEng_ReduceAlong(t *testing.T) {
	assert := assert.New(t)

	a := dense.New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
	eng := StdEng[float64, *dense.Dense[float64]]{}
	ctx := context.Background()

	module := tensor.ReductionModule[float64]{
		MonotonicReduction: func(a []float64) float64 {
			var retVal float64
			for i := range a {
				retVal += a[i]
			}
			return retVal
		},
		ReduceFirstN: func(retVal, a []float64) {
			for i := range retVal {
				retVal[i] += a[i]
			}
		},
		ReduceLastN: func(a []float64, defaultValue float64) float64 {
			retVal := defaultValue
			for i := range a {
				retVal += a[i]
			}
			return retVal
		},
		Reduce: func(a, b float64) float64 { return a + b },
	}

	// Sum over axes 0 and 2
	retVal := dense.New[float64](WithShape(3, 4))
	err := eng.ReduceAlong(ctx, module, 0.0, a, retVal, 0, 2)
	assert.NoError(err)
	assert.Equal([]float64{60, 92, 124}, retVal.Data())
	assert.True(shapes.Shape{3}.Eq(retVal.Shape()))

	// Sum over axes 2 and 0
	retVal = dense.New[float64](WithShape(3, 4))
	err = eng.ReduceAlong(ctx, module, 0.0, a, retVal, 2, 0)
	assert.NoError(err)
	assert.Equal([]float64{60, 92, 124}, retVal.Data())
	assert.True(shapes.Shape{3}.Eq(retVal.Shape()))

	// Sum over axes 1 and 2
	retVal = dense.New[float64](WithShape(2, 4))
	err = eng.ReduceAlong(ctx, module, 0.0, a, retVal, 1, 2)
	assert.NoError(err)
	assert.Equal([]float64{66, 210}, retVal.Data())

	// Reduce over no axes (aka monotonic sum)
	retVal = dense.New[float64](WithShape())
	err = eng.ReduceAlong(ctx, module, 0.0, a, retVal)
	assert.NoError(err)
	assert.Equal(module.MonotonicReduction(gutils.Range[float64](0, 24)), retVal.ScalarValue())
	assert.True(shapes.ScalarShape().Eq(retVal.Shape()))
}

func ExampleStdEng_ReduceAlong_nonStdReduction() {
	minus := func(a, b float32) float32 { return a - b }
	a := dense.New[float32](WithShape(2, 3), WithBacking([]float32{
		5, 10, -2,
		-6, -2, 9,
	}))
	eng := StdEng[float32, *dense.Dense[float32]]{}
	ctx := context.Background()
	fmt.Printf("a:\n%v\n", a)

	// We want to reduce along axes []int{1, 0}
	fmt.Println("Reduce a along []int{1, 0}\n--------------------------")
	// set up
	var err error
	retVal := dense.New[float32](WithShape(2, 3)) // overprovisioned retVal
	if err = eng.Reduce(ctx, minus, a, 1, 0.0, retVal); err != nil {
		fmt.Printf("err %v\n", err)
	}
	fmt.Printf("After reducing along axis 1:\n%v\n", retVal)

	if err = eng.Reduce(ctx, minus, retVal, 0, 0.0, retVal); err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("After reducing along axis 0:\n%v\n\n", retVal)

	// ReduceAlong:
	fmt.Println("Using ReduceAlong along []int{1,0}:\n-----------------------------------")
	mod := tensor.ReductionModule[float32]{Reduce: minus, IsNonCommutative: true}
	retVal = dense.New[float32](WithShape(2, 3))
	if err = eng.ReduceAlong(ctx, mod, 0, a, retVal, 1, 0); err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("After ReduceAlong:\n%v\n\n=====================================\n\n", retVal)

	// What would happen if we reduce it along []int{0, 1}
	fmt.Println("What would happen if we reduce along []int{0,1}\n-----------------------------------------------")
	retVal = dense.New[float32](WithShape(2, 3)) // overprovisioned retVal
	if err = eng.Reduce(ctx, minus, a, 0, 0.0, retVal); err != nil {
		fmt.Printf("err %v\n", err)
	}
	fmt.Printf("After reducing along axis 0:\n%v\n", retVal)

	if err = eng.Reduce(ctx, minus, retVal, 0, 0.0, retVal); err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("After reducing along axis 1:\n%v\n\n", retVal)

	fmt.Println("Using ReduceAlong along []int{0,1}:\n-----------------------------------")
	retVal = dense.New[float32](WithShape(2, 3))
	if err = eng.ReduceAlong(ctx, mod, 0, a, retVal, 0, 1); err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("After ReduceAlong:\n%v\n\n", retVal)
	fmt.Println("Be careful to mark non-commutative functions. If `IsNonCommutative` is false:")
	mod.IsNonCommutative = false
	if err = eng.ReduceAlong(ctx, mod, 0, a, retVal, 0, 1); err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("After ReduceAlong []int{0, 1} (INCORRECT):\n%v\n\n", retVal)

	// Output:
	// a:
	// ⎡ 5  10  -2⎤
	// ⎣-6  -2   9⎦
	//
	// Reduce a along []int{1, 0}
	// --------------------------
	// After reducing along axis 1:
	// [-13   -1]
	// After reducing along axis 0:
	// -12
	//
	// Using ReduceAlong along []int{1,0}:
	// -----------------------------------
	// After ReduceAlong:
	// -12
	//
	// =====================================
	//
	// What would happen if we reduce along []int{0,1}
	// -----------------------------------------------
	// After reducing along axis 0:
	// [ 11   12  -11]
	// After reducing along axis 1:
	// 10
	//
	// Using ReduceAlong along []int{0,1}:
	// -----------------------------------
	// After ReduceAlong:
	// 10
	//
	// Be careful to mark non-commutative functions. If `IsNonCommutative` is false:
	// After ReduceAlong []int{0, 1} (INCORRECT):
	// -14

}

func TestPrepReduce(t *testing.T) {
	assert := assert.New(t)

	t.Run("Safe, Valid axes", func(t *testing.T) {
		a := dense.New[float64](WithShape(2, 3, 4),
			WithBacking([]float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}))

		e := StdEng[float64, *dense.Dense[float64]]{}
		_, axes, retVal, err := e.PrepReduce(a, dense.Along(1, 2))

		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		assert.Equal([]int{1, 2}, axes)
		assert.NotNil(retVal)
		assert.True(retVal.Shape().Eq(shapes.Shape{3, 4}), "Expected %v. Got %v instead", shapes.Shape{3, 4}, retVal.Shape())
	})

	t.Run("Safe, invalid axes", func(t *testing.T) {
		a := dense.New[float64](WithShape(2, 3, 4),
			WithBacking([]float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}))
		e := StdEng[float64, *dense.Dense[float64]]{}
		_, _, _, err := e.PrepReduce(a, tensor.Along(1, 3))
		assert.NotNil(err)
	})

	t.Run("Safe, no axes", func(t *testing.T) {
		a := dense.New[float64](WithShape(2, 3, 4),
			WithBacking([]float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}))
		e := StdEng[float64, *dense.Dense[float64]]{}
		_, axes, retVal, err := e.PrepReduce(a)
		assert.Nil(err)
		assert.Equal([]int{0, 1, 2}, axes)
		assert.NotNil(retVal)
		assert.True(retVal.Shape().Eq(shapes.Shape{3, 4}))

	})

	t.Run("With Good Reuse", func(t *testing.T) {
		a := dense.New[float64](WithShape(2, 3, 4),
			WithBacking([]float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}))
		r := dense.New[float64](WithShape(2, 3, 4))
		e := StdEng[float64, *dense.Dense[float64]]{}
		_, axes, retVal, err := e.PrepReduce(a, tensor.WithReuse(r))
		assert.Nil(err)
		assert.Equal([]int{0, 1, 2}, axes)
		assert.NotNil(retVal)
		assert.True(retVal.Shape().Eq(shapes.Shape{3, 4}))

	})

	t.Run("With Bad Reuse", func(t *testing.T) {
		a := dense.New[float64](WithShape(2, 3, 4),
			WithBacking([]float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}))
		r := dense.New[float64](WithShape(3, 3))
		e := StdEng[float64, *dense.Dense[float64]]{}
		_, _, _, err := e.PrepReduce(a, tensor.WithReuse(r))
		assert.NotNil(err)

	})

}

type concatTest[DT any] struct {
	name   string
	a      *dense.Dense[DT]
	others []*dense.Dense[DT]
	axis   int

	correctShape shapes.Shape
	correctData  []DT
}

func makeConcatTests[DT internal.Num]() []concatTest[DT] {
	return []concatTest[DT]{
		{"vector-vector",
			dense.New[DT](WithShape(2), WithBacking([]DT{1, 2})),
			[]*dense.Dense[DT]{dense.New[DT](WithShape(2), WithBacking([]DT{3, 4}))},
			0, shapes.Shape{4}, []DT{1, 2, 3, 4}},
		{"vector-vector (many)",
			dense.New[DT](WithShape(2), WithBacking([]DT{1, 2})),
			[]*dense.Dense[DT]{
				dense.New[DT](WithShape(2), WithBacking([]DT{3, 4})),
				dense.New[DT](WithShape(2), WithBacking([]DT{5, 6})),
			}, 0, shapes.Shape{6}, []DT{1, 2, 3, 4, 5, 6}},
		{"matrix-matrix, axis 0",
			dense.New[DT](WithShape(2, 2), WithBacking([]DT{
				1, 2,
				3, 4})),
			[]*dense.Dense[DT]{dense.New[DT](WithShape(2, 2), WithBacking([]DT{
				5, 6,
				7, 8}))},
			0,
			shapes.Shape{4, 2},
			[]DT{
				1, 2,
				3, 4,
				5, 6,
				7, 8},
		},

		{"matrix-matrix, axis 1",
			dense.New[DT](WithShape(2, 2), WithBacking([]DT{
				1, 2,
				3, 4})),
			[]*dense.Dense[DT]{dense.New[DT](WithShape(2, 2), WithBacking([]DT{
				5, 6,
				7, 8}))},
			1,
			shapes.Shape{2, 4},
			[]DT{
				1, 2, 5, 6,
				3, 4, 7, 8},
		},

		{"3tensor-3tensor, axis 0",
			dense.New[DT](WithShape(2, 2, 2), WithBacking([]DT{
				1, 2,
				3, 4,

				5, 6,
				7, 8})),
			[]*dense.Dense[DT]{dense.New[DT](WithShape(2, 2, 2), WithBacking([]DT{
				9, 10,
				11, 12,

				13, 14,
				15, 16}))},
			0,
			shapes.Shape{4, 2, 2},
			[]DT{
				1, 2,
				3, 4,

				5, 6,
				7, 8,

				9, 10,
				11, 12,

				13, 14,
				15, 16},
		},
		{"3tensor-3tensor, axis 1",
			dense.New[DT](WithShape(2, 2, 2), WithBacking([]DT{
				1, 2,
				3, 4,

				5, 6,
				7, 8})),
			[]*dense.Dense[DT]{dense.New[DT](WithShape(2, 2, 2), WithBacking([]DT{
				9, 10,
				11, 12,

				13, 14,
				15, 16}))},
			1,
			shapes.Shape{2, 4, 2},
			[]DT{
				1, 2,
				3, 4,
				9, 10,
				11, 12,

				5, 6,
				7, 8,
				13, 14,
				15, 16},
		},
		{"3tensor-3tensor, axis 2",
			dense.New[DT](WithShape(2, 2, 2), WithBacking([]DT{
				1, 2,
				3, 4,

				5, 6,
				7, 8})),
			[]*dense.Dense[DT]{dense.New[DT](WithShape(2, 2, 2), WithBacking([]DT{
				9, 10,
				11, 12,

				13, 14,
				15, 16}))},
			2,
			shapes.Shape{2, 2, 4},
			[]DT{
				1, 2, 9, 10,
				3, 4, 11, 12,

				5, 6, 13, 14,
				7, 8, 15, 16},
		},
	}
}

func TestDense_Concat(t *testing.T) {
	assert := assert.New(t)
	for _, tc := range makeConcatTests[float64]() {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.a.Concat(tc.axis, tc.others...)
			if err != nil {
				t.Errorf("Error: %v", err)
				return
			}

			assert.Equal(tc.correctShape, got.Shape(), "%v failed. Wrong resulting shape", tc.name)
			assert.Equal(tc.correctData, got.Data(), "%v failed. Wrong resulting data", tc.name)

		})

	}
}
