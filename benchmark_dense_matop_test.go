package tensor

import (
	"math/rand"
	"testing"
)

func BenchmarkDense_Transpose(b *testing.B) {
	T := New(WithShape(100, 100, 2), WithBacking(Range(Byte, 0, 100*100*2)))
	for i := 0; i < b.N; i++ {
		T.T()
		T.Transpose()
	}
}

func BenchmarkNativeSet(b *testing.B) {
	T := New(WithShape(10000), Of(Float64))
	data := T.Data().([]float64)
	for i := 0; i < b.N; i++ {
		for next := 0; next < 10000; next++ {
			data[next] = float64(next + 1)
		}
	}
}

func BenchmarkSetMethod(b *testing.B) {
	T := New(WithShape(10000), Of(Float64))
	for i := 0; i < b.N; i++ {
		for next := 0; next < 10000; next++ {
			T.Set(next, float64(next+1))
		}
	}
}

func BenchmarkNativeGet(b *testing.B) {
	T := New(WithShape(10000), Of(Float64))
	data := T.Data().([]float64)
	var f float64
	for i := 0; i < b.N; i++ {
		for next := 0; next < 10000; next++ {
			f = data[next]
		}
	}
	_ = f
}

func BenchmarkGetMethod(b *testing.B) {
	T := New(WithShape(10000), Of(Float64))
	var f float64
	for i := 0; i < b.N; i++ {
		for next := 0; next < 10000; next++ {
			f = T.Get(next).(float64)
		}
	}
	_ = f
}

func BenchmarkGetWithIterator(b *testing.B) {
	T := New(WithShape(100, 100), Of(Float64))
	var f float64
	data := T.Data().([]float64)
	for i := 0; i < b.N; i++ {
		it := IteratorFromDense(T)
		var next int
		var err error
		for next, err = it.Start(); err == nil; next, err = it.Next() {
			f = data[next]
		}
		if _, ok := err.(NoOpError); !ok {
			b.Errorf("Error: %v", err)
		}
	}
	_ = f
}

func BenchmarkComplicatedGet(b *testing.B) {
	T := New(WithShape(101, 1, 36, 5), Of(Float64))
	T.T(0, 2, 1, 3)
	data := T.Data().([]float64)
	var f float64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		it := IteratorFromDense(T)
		var next int

		var err error
		for next, err = it.Start(); err == nil; next, err = it.Next() {
			f = data[next]
		}
		if _, ok := err.(NoOpError); !ok {
			b.Errorf("Error: %v", err)
		}
	}
	_ = f
}

var atCoords [10000][2]int

func init() {
	for i := range atCoords {
		atCoords[i][0] = rand.Intn(100)
		atCoords[i][1] = rand.Intn(100)
	}
}

var at1, at2 float64

// func BenchmarkAtWithNativeIterator(b *testing.B) {
// 	T := New(WithShape(100, 100), Of(Float64))
// 	it, err := NativeMatrixF64(T)
// 	if err != nil {
// 		b.Fatalf("Error: %v", err)
// 	}

// 	var j int
// 	for i := 0; i < b.N; i++ {

// 		if j >= len(atCoords) {
// 			j = 0
// 		}

// 		at := atCoords[j]
// 		at1 = it[at[0]][at[1]]
// 		j++
// 	}
// }

func BenchmarkAt(b *testing.B) {
	T := New(WithShape(100, 100), Of(Float64))
	var j int
	for i := 0; i < b.N; i++ {
		if j >= len(atCoords) {
			j = 0
		}

		at := atCoords[j]
		_, err := T.At(at[0], at[1])
		if err != nil {
			b.Errorf("Error: %v", err)
		}

		j++
	}
}
