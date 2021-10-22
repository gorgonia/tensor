package tensor

import "fmt"

func ExampleByIndices() {
	a := New(WithShape(2, 2), WithBacking([]float64{
		100, 200,
		300, 400,
	}))
	indices := New(WithBacking([]int{1, 1, 1, 0, 1}))
	b, err := ByIndices(a, indices, 0) // we select rows 1, 1, 1, 0, 1
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("a:\n%v\nindices: %v\nb:\n%v\n", a, indices, b)

	// Output:
	// a:
	// ⎡100  200⎤
	// ⎣300  400⎦
	//
	// indices: [1  1  1  0  1]
	// b:
	// ⎡300  400⎤
	// ⎢300  400⎥
	// ⎢300  400⎥
	// ⎢100  200⎥
	// ⎣300  400⎦

}

func ExampleByIndicesB() {
	a := New(WithShape(2, 2), WithBacking([]float64{
		100, 200,
		300, 400,
	}))
	indices := New(WithBacking([]int{1, 1, 1, 0, 1}))
	b, err := ByIndices(a, indices, 0) // we select rows 1, 1, 1, 0, 1
	if err != nil {
		fmt.Println(err)
		return
	}

	outGrad := b.Clone().(*Dense)
	outGrad.Memset(1.0)

	grad, err := ByIndicesB(a, outGrad, indices, 0)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("a:\n%v\nindices: %v\nb:\n%v\ngrad:\n%v", a, indices, b, grad)

	// Output:
	// a:
	// ⎡100  200⎤
	// ⎣300  400⎦
	//
	// indices: [1  1  1  0  1]
	// b:
	// ⎡300  400⎤
	// ⎢300  400⎥
	// ⎢300  400⎥
	// ⎢100  200⎥
	// ⎣300  400⎦
	//
	// grad:
	// ⎡1  1⎤
	// ⎣4  4⎦

}
