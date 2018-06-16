package tensor

import (
	"fmt"
)

func ExampleDense_MatMul() {
	handleErr := func(err error) {
		if err != nil {
			panic(err)
		}
	}

	T0 := New(WithShape(10, 15), WithBacking(Range(Float64, 0, 150)))
	T1 := New(WithShape(15, 10), WithBacking(Range(Float64, 150, 0)))
	T2, err := MatMul(T0, T1)
	handleErr(err)

	fmt.Printf("T2:\n%v", T2)

	// Output:
	// T2:
	// ⎡  5600    5495    5390    5285  ...   4970    4865    4760    4655⎤
	// ⎢ 23600   23270   22940   22610  ...  21620   21290   20960   20630⎥
	// ⎢ 41600   41045   40490   39935  ...  38270   37715   37160   36605⎥
	// ⎢ 59600   58820   58040   57260  ...  54920   54140   53360   52580⎥
	// .
	// .
	// .
	// ⎢113600  112145  110690  109235  ... 104870  103415  101960  100505⎥
	// ⎢131600  129920  128240  126560  ... 121520  119840  118160  116480⎥
	// ⎢149600  147695  145790  143885  ... 138170  136265  134360  132455⎥
	// ⎣167600  165470  163340  161210  ... 154820  152690  150560  148430⎦

}

func ExampleDense_MatVecMul() {
	handleErr := func(err error) {
		if err != nil {
			panic(err)
		}
	}

	T0 := New(WithShape(2, 3), WithBacking(Range(Float64, 1, 7)))
	T1 := New(WithShape(3), WithBacking(Range(Float64, 0, 3)))
	T2, err := T0.MatVecMul(T1)
	handleErr(err)

	fmt.Printf("T2:\n%v\n", T2)

	// Output:
	// T2:
	// [ 8  17]
}

func ExampleDense_MatVecMul_rowMajorSliced() {
	// ASPIRATIONAL TODO: IncX and incY of differering values

	handleErr := func(err error) {
		if err != nil {
			panic(err)
		}
	}

	T0 := New(WithShape(10, 12), WithBacking(Range(Float64, 1, 121)))
	T1 := New(WithShape(3, 3), WithBacking(Range(Float64, 1, 10)))
	T2, err := T0.Slice(makeRS(1, 3), makeRS(3, 6))
	handleErr(err)
	T3, err := T1.Slice(nil, makeRS(1, 2))
	handleErr(err)

	// here the + formatting option is used because you should know that after this particular slice, the result will be a vector
	fmt.Printf("T2:\n%+v", T2)
	fmt.Printf("T3:\n%+v\n", T3)

	// here we print the underlying slice of T3 just to show that it's actually a much larger slice
	fmt.Printf("Underlying Slice: %v\n", T3.Data())

	T4, err := T2.(*Dense).MatVecMul(T3)
	handleErr(err)

	fmt.Printf("T4:\n%v\n", T4)

	// Outputz:
	// T2:
	// Matrix (2, 3) [10 1]
	// ⎡14  15  16⎤
	// ⎣24  25  26⎦
	// T3:
	// Vector (3) [3]
	// [2  5  8]
	// Underlying Slice: [2 3 4 5 6 7 8]
	// T4:
	// [261  441]

}

func ExampleDense_MatMul_sliced() {
	//ASPIRATIONAL TODO: incX and incY of different sizes
	handleErr := func(err error) {
		if err != nil {
			panic(err)
		}
	}

	T0 := New(WithShape(10, 15), WithBacking(Range(Float64, 0, 150)))
	T1 := New(WithShape(15, 10), WithBacking(Range(Float64, 150, 0)))
	T2, err := MatMul(T0, T1)
	handleErr(err)

	fmt.Printf("T2:\n%v", T2)

	// Slice T0 to only take a (2, 3) on the upper quadrant
	// T3 := T0[0:3, 0:2]
	T3, err := T0.Slice(makeRS(0, 3), makeRS(0, 2))
	handleErr(err)
	fmt.Printf("T3:\n%v", T3)

	T4, err := T1.Slice(makeRS(13, 15), makeRS(8, 10))
	handleErr(err)
	fmt.Printf("T4:\n%v", T4)

	T5, err := T3.(*Dense).MatMul(T4)
	handleErr(err)
	fmt.Printf("T3xT4:\n%v", T5)

	// Outputz:
	// T2:
	// ⎡  5600    5495    5390    5285  ...   4970    4865    4760    4655⎤
	// ⎢ 23600   23270   22940   22610  ...  21620   21290   20960   20630⎥
	// ⎢ 41600   41045   40490   39935  ...  38270   37715   37160   36605⎥
	// ⎢ 59600   58820   58040   57260  ...  54920   54140   53360   52580⎥
	// .
	// .
	// .
	// ⎢113600  112145  110690  109235  ... 104870  103415  101960  100505⎥
	// ⎢131600  129920  128240  126560  ... 121520  119840  118160  116480⎥
	// ⎢149600  147695  145790  143885  ... 138170  136265  134360  132455⎥
	// ⎣167600  165470  163340  161210  ... 154820  152690  150560  148430⎦
	// T3:
	// ⎡ 0   1⎤
	// ⎢15  16⎥
	// ⎣30  31⎦
	// T4:
	// ⎡12  11⎤
	// ⎣ 2   1⎦
	// T3xT4:
	// ⎡  2    1⎤
	// ⎢212  181⎥
	// ⎣422  361⎦
}
