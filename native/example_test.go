package native

import (
	"fmt"

	. "gorgonia.org/tensor"
)

// There are times where it is more effective to use native Go slice semantics to do work (for example, when performing batch work over kernels).
// Iterators are useful for this purpose. This package provides iterators for the standard types
// However, custom types are also available. See Vector, Matrix and Tensor3 examples.
func Example_iterator() {
	var T *Dense
	T = New(WithShape(2, 3), WithBacking(Range(Float64, 0, 6)))
	x, err := MatrixF64(T)
	if err != nil {
		fmt.Printf("ERR: %v", err)
	}

	for _, row := range x {
		fmt.Printf("%v\n", row)
	}

	// Output:
	// [0 1 2]
	// [3 4 5]
}

// The NativeSelect function squashes the dimensions, and returns an iterator in native Go slice semantics.
func Example_select() {
	// Selection is a bit of an interesting use case. Sometimes you don't want to iterate through the layers.
	//
	// For example, in a number of use cases where you have a 4-Tensor, you'd typically reshape it to some
	// 2D matrix which can then be plugged into BLAS algorithms directly. Sometimes you wouldn't need to reshape.
	// All you have to do is squash the dimensions inwards. This function does that.
	//
	// The best way to explain the Select functions is through concrete examples.
	// Imagine a tensor with (2,3,4,5) shape. Arbitrarily, we call them (NCHW) - Batch Size, Channel Count, Height, Width.
	// If we want to select all the channels, across all batches, then `NativeSelectX(T, 1)` would yield all channels. The resulting matrix will be (6, 20)
	// If we want to select all the heights, across all channels and batches, then `NativeSelectX(T, 2) will yield all heights. The resulting matrix will be (24, 5)
	//
	// If for some reason the format was in NHWC, then you would need to reshape. This wouldn't be useful.

	var T *Dense
	T = New(WithShape(2, 3, 4, 5), WithBacking(Range(Float64, 0, 2*3*4*5)))
	x, err := SelectF64(T, 1)
	if err != nil {
		fmt.Printf("ERR %v", err)
	}
	for _, row := range x {
		fmt.Printf("%3.0f\n", row)
	}

	// Output:
	// [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19]
	// [ 20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39]
	// [ 40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59]
	// [ 60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79]
	// [ 80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99]
	// [100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119]
}

// The iterators are iteratos in the truest sense. The data isn't copied, as this example shows
func Example_clobber() {
	var T *Dense
	T = New(WithShape(2, 3), WithBacking(Range(Float64, 0, 6)))
	fmt.Printf("Before :\n%v", T)

	xx, _ := MatrixF64(T)
	xx[1][1] = 10000
	fmt.Printf("After :\n%v", T)

	// Output:
	// Before :
	// ⎡0  1  2⎤
	// ⎣3  4  5⎦
	// After :
	// ⎡    0      1      2⎤
	// ⎣    3  10000      5⎦

}
