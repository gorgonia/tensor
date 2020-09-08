package tensor_test

import (
	"fmt"

	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/memory"
	"gorgonia.org/tensor"
)

func ExampleDense_Arrow() {
	pool := memory.NewGoAllocator()

	b := array.NewFloat64Builder(pool)
	defer b.Release()

	b.AppendValues(
		[]float64{1, 2, 3, -1, 4, 5},
		[]bool{true, true, true, false, true, true},
	)

	arr := b.NewFloat64Array()
	defer arr.Release()
	fmt.Printf("arrow array = %v\n", arr)

	a := tensor.FromArrowArray(arr)
	fmt.Printf("tensor = %v\n", a)

	// Output:
	// arrow array = [1 2 3 (null) 4 5]
	// tensor = C[ 1   2   3  --   4   5]
}
