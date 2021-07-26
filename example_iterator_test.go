package tensor

import (
	"fmt"
	"sync"
)

// This is an example of how to use `IteratorFromDense` from a row-major Dense tensor
func Example_iteratorRowmajor() {
	T := New(WithShape(2, 3), WithBacking([]float64{0, 1, 2, 3, 4, 5}))
	it := IteratorFromDense(T)
	fmt.Printf("T:\n%v\n", T)

	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i: %d, coord: %v\n", i, it.Coord())
	}

	// Output:
	// T:
	// ⎡0  1  2⎤
	// ⎣3  4  5⎦
	//
	// i: 0, coord: [0 1]
	// i: 1, coord: [0 2]
	// i: 2, coord: [1 0]
	// i: 3, coord: [1 1]
	// i: 4, coord: [1 2]
	// i: 5, coord: [0 0]

}

// This is an example of using `IteratorFromDense` on a col-major Dense tensor. More importantly
// this example shows the order of the iteration.
func Example_iteratorcolMajor() {
	T := New(WithShape(2, 3), WithBacking([]float64{0, 1, 2, 3, 4, 5}), AsFortran(nil))
	it := IteratorFromDense(T)
	fmt.Printf("T:\n%v\n", T)

	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i: %d, coord: %v\n", i, it.Coord())
	}

	// Output:
	// T:
	// ⎡0  2  4⎤
	// ⎣1  3  5⎦
	//
	// i: 0, coord: [0 1]
	// i: 2, coord: [0 2]
	// i: 4, coord: [1 0]
	// i: 1, coord: [1 1]
	// i: 3, coord: [1 2]
	// i: 5, coord: [0 0]

}

func ExampleSliceIter() {
	T := New(WithShape(3, 3), WithBacking(Range(Float64, 0, 9)))
	S, err := T.Slice(makeRS(1, 3), makeRS(1, 3))
	if err != nil {
		fmt.Printf("Err %v\n", err)
		return
	}
	fmt.Printf("S (requires iterator? %t)\n%v\n", S.(DenseView).RequiresIterator(), S)
	it := IteratorFromDense(S.(DenseView))
	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i %d, coord %v\n", i, it.Coord())
	}

	// Output:
	// S (requires iterator? true)
	// ⎡4  5⎤
	// ⎣7  8⎦
	//
	// i 0, coord [0 1]
	// i 1, coord [1 0]
	// i 3, coord [1 1]
	// i 4, coord [0 0]

}

func ExampleAxialIterator() {
	T := New(WithShape(2, 3, 4), WithBacking([]float64{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
	}))
	fmt.Printf("T:\n%v", T)
	it := AxialIteratorFromDense(T, 1, 0, false)

	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i %d coord %v\n", i, it.Coord())
	}

	// Output:
	// T:
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎣ 8   9  10  11⎦
	//
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎣ 8   9  10  11⎦
	//
	// i 0 coord [0 0 1]
	// i 1 coord [0 0 2]
	// i 2 coord [0 0 3]
	// i 3 coord [1 0 0]
	// i 12 coord [1 0 1]
	// i 13 coord [1 0 2]
	// i 14 coord [1 0 3]
	// i 15 coord [0 1 0]
	// i 4 coord [0 1 1]
	// i 5 coord [0 1 2]
	// i 6 coord [0 1 3]
	// i 7 coord [1 1 0]
	// i 16 coord [1 1 1]
	// i 17 coord [1 1 2]
	// i 18 coord [1 1 3]
	// i 19 coord [0 2 0]
	// i 8 coord [0 2 1]
	// i 9 coord [0 2 2]
	// i 10 coord [0 2 3]
	// i 11 coord [1 2 0]
	// i 20 coord [1 2 1]
	// i 21 coord [1 2 2]
	// i 22 coord [1 2 3]
	// i 23 coord [0 0 0]
}

func ExampleAxialIterator_2() {
	T := New(WithShape(2, 3, 4), WithBacking([]float64{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
	}))
	fmt.Printf("T:\n%v", T)
	it := AxialIteratorFromDense(T, 1, 1, true)

	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i %d coord %v\n", i, it.Coord())
	}

	// Output:
	// T:
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎣ 8   9  10  11⎦
	//
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎣ 8   9  10  11⎦
	//
	// i 4 coord [0 1 1]
	// i 5 coord [0 1 2]
	// i 6 coord [0 1 3]
	// i 7 coord [1 1 0]
	// i 16 coord [1 1 1]
	// i 17 coord [1 1 2]
	// i 18 coord [1 1 3]
	// i 19 coord [0 0 0]
}

func ExampleAxialIterator_concurrent() {
	T := New(WithShape(2, 3, 4), WithBacking([]float64{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
	}))
	fmt.Printf("T:\n%v", T)

	axis := 1
	var its []Iterator
	for i := 0; i < T.Shape()[axis]; i++ {
		it := AxialIteratorFromDense(T, axis, i, true)
		its = append(its, it)
	}

	done := make(chan float64, T.Shape()[axis])
	var wg sync.WaitGroup
	for _, it := range its {
		wg.Add(1)
		go func(it Iterator, t *Dense, done chan float64, wg *sync.WaitGroup) {
			data := t.Data().([]float64)
			var sum float64
			for i, err := it.Start(); err == nil; i, err = it.Next() {
				sum += data[i]
			}
			done <- sum
			wg.Done()
		}(it, T, done, &wg)
	}

	wg.Wait()
	close(done)

	var total float64
	for v := range done {
		total += v
	}

	fmt.Printf("Total: %v", total)

	// Output:
	// T:
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎣ 8   9  10  11⎦
	//
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎣ 8   9  10  11⎦
	//
	// Total: 132

}
