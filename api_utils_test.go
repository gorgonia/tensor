package tensor

import (
	"testing"
)

type testInt []int

func (m testInt) Less(i, j int) bool { return m[i] < m[j] }
func (m testInt) Len() int           { return len(m) }
func (m testInt) Swap(i, j int)      { m[i], m[j] = m[j], m[i] }

func TestSortIndexInts(t *testing.T) {
	in := []int{9, 8, 7, 6, 5, 4, 10, -1, -2, -4, 11, 13, 15, 100, 99}
	inCopy := make([]int, len(in))
	copy(inCopy, in)
	out := SortIndex(in)
	for i := 1; i < len(out); i++ {
		if inCopy[out[i]] < inCopy[out[i-1]] {
			t.Fatalf("Unexpected output")
		}
	}
	for i := range in {
		if in[i] != inCopy[i] {
			t.Fatalf("The input slice should not be changed")
		}
	}
}

func TestSortIndexFloats(t *testing.T) {
	in := []float64{.9, .8, .7, .6, .5, .4, .10, -.1, -.2, -.4, .11, .13, .15, .100, .99}
	inCopy := make([]float64, len(in))
	copy(inCopy, in)
	out := SortIndex(in)
	for i := 1; i < len(out); i++ {
		if inCopy[out[i]] < inCopy[out[i-1]] {
			t.Fatalf("Unexpected output")
		}
	}
	for i := range in {
		if in[i] != inCopy[i] {
			t.Fatalf("The input slice should not be changed")
		}
	}
}

func TestSortIndexSortInterface(t *testing.T) {
	in := testInt{9, 8, 7, 6, 5, 4, 10, -1, -2, -4, 11, 13, 15, 100, 99}
	inCopy := make(testInt, len(in))
	copy(inCopy, in)
	out := SortIndex(in)
	for i := 1; i < len(out); i++ {
		if inCopy[out[i]] < inCopy[out[i-1]] {
			t.Fatalf("Unexpected output")
		}
	}
	for i := range in {
		if in[i] != inCopy[i] {
			t.Fatalf("The input slice should not be changed")
		}
	}
}
