package tensor

import (
	"math"
	"math/rand"
	"reflect"
	"sort"

	"github.com/chewxy/math32"
)

// SortIndex: Similar to numpy's argsort.
// Returns indices for sorting a slice in increasing order.
// Input slice remains unchanged.
// SortIndex may not be stable; for stability, use SortIndexStable.
func SortIndex(in interface{}) (out []int) {
	return sortIndex(in, sort.Slice)
}

// SortIndexStable: Similar to SortIndex, but stable.
// Returns indices for sorting a slice in increasing order.
// Input slice remains unchanged.
func SortIndexStable(in interface{}) (out []int) {
	return sortIndex(in, sort.SliceStable)
}

func sortIndex(in interface{}, sortFunc func(x interface{}, less func(i int, j int) bool)) (out []int) {
	switch list := in.(type) {
	case []int:
		out = make([]int, len(list))
		for i := 0; i < len(list); i++ {
			out[i] = i
		}
		sortFunc(out, func(i, j int) bool {
			return list[out[i]] < list[out[j]]
		})
	case []float64:
		out = make([]int, len(list))
		for i := 0; i < len(list); i++ {
			out[i] = i
		}
		sortFunc(out, func(i, j int) bool {
			return list[out[i]] < list[out[j]]
		})
	case sort.Interface:
		out = make([]int, list.Len())
		for i := 0; i < list.Len(); i++ {
			out[i] = i
		}
		sortFunc(out, func(i, j int) bool {
			return list.Less(out[i], out[j])
		})
	default:
		panic("The slice type is not currently supported.")
	}

	return
}

// SampleIndex samples a slice or a Tensor.
// TODO: tidy this up.
func SampleIndex(in interface{}) int {
	// var l int
	switch list := in.(type) {
	case []int:
		var sum, i int
		// l = len(list)
		r := rand.Int()
		for {
			sum += list[i]
			if sum > r && i > 0 {
				return i
			}
			i++
		}
	case []float64:
		var sum float64
		var i int
		// l = len(list)
		r := rand.Float64()
		for {
			sum += list[i]
			if sum > r && i > 0 {
				return i
			}
			i++
		}
	case *Dense:
		var i int
		switch list.t.Kind() {
		case reflect.Float64:
			var sum float64
			r := rand.Float64()
			data := list.Float64s()
			// l = len(data)
			for {
				datum := data[i]
				if math.IsNaN(datum) || math.IsInf(datum, 0) {
					return i
				}

				sum += datum
				if sum > r && i > 0 {
					return i
				}
				i++
			}
		case reflect.Float32:
			var sum float32
			r := rand.Float32()
			data := list.Float32s()
			// l = len(data)
			for {
				datum := data[i]
				if math32.IsNaN(datum) || math32.IsInf(datum, 0) {
					return i
				}

				sum += datum
				if sum > r && i > 0 {
					return i
				}
				i++
			}
		default:
			panic("not yet implemented")
		}
	default:
		panic("Not yet implemented")
	}
	return -1
}
