package execution

import (
	"fmt"
	"reflect"
	"testing"
)

func TestLtVV(t *testing.T) {
	t.Run("int", func(t *testing.T) {
		a := []int{1, 2, 3}
		b := []int{2, 1, 4}
		c := make([]int, len(a))
		LtVV(a, b, c)
		expected := []int{1, 0, 1}
		if !reflect.DeepEqual(c, expected) {
			t.Errorf("Expected %v, got %v", expected, c)
		}
	})

	t.Run("float64", func(t *testing.T) {
		a := []float64{1.1, 2.2, 3.3}
		b := []float64{2.2, 2.1, 4.4}
		c := make([]float64, len(a))
		LtVV(a, b, c)
		expected := []float64{1, 0, 1}
		if !reflect.DeepEqual(c, expected) {
			t.Errorf("Expected %v, got %v", expected, c)
		}
	})

	t.Run("equal", func(t *testing.T) {
		a := []int{1, 2, 3}
		b := []int{1, 2, 3}
		c := make([]int, len(a))
		LtVV(a, b, c)
		expected := []int{0, 0, 0}
		if !reflect.DeepEqual(c, expected) {
			t.Errorf("Expected %v, got %v", expected, c)
		}
	})
}
func TestLtVVIter(t *testing.T) {
	a := []int{1, 2, 3}
	b := []int{2, 1, 4}
	c := []int{0, 0, 0}

	ait := newMockIterator[int](a)
	bit := newMockIterator[int](b)
	cit := newMockIterator[int](c)

	err := LtVVIter(a, b, c, ait, bit, cit)
	if err != nil {
		t.Fatal(err)
	}

	expected := []int{1, 0, 1}
	if !reflect.DeepEqual(c, expected) {
		t.Errorf("Expected %v, got %v", expected, c)
	}
}

func TestLtVS(t *testing.T) {
	tests := []struct {
		a        []int
		b        int
		expected []int
	}{
		{
			a:        []int{1, 2, 3},
			b:        2,
			expected: []int{1, 0, 0},
		},
		{
			a:        []int{1, 3, 2},
			b:        2,
			expected: []int{1, 0, 0},
		},
		{
			a:        []int{2, 2, 2},
			b:        1,
			expected: []int{0, 0, 0},
		},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%v %v", test.a, test.b), func(t *testing.T) {
			c := make([]int, len(test.a))
			LtVS(test.a, test.b, c)
			if !reflect.DeepEqual(c, test.expected) {
				t.Errorf("Expected %v, got %v", test.expected, c)
			}
		})
	}
}

func TestLtVSIter(t *testing.T) {
	a := []int{1, 2, 3}
	b := 2
	c := []int{0, 0, 0}

	ait := newMockIterator(a)
	cit := newMockIterator(c)

	err := LtVSIter(a, b, c, ait, cit)
	if err != nil {
		t.Fatal(err)
	}

	expected := []int{1, 0, 0}
	if !reflect.DeepEqual(c, expected) {
		t.Errorf("Expected %v, got %v", expected, c)
	}

	// when the return slice is the input slice
	ait = newMockIterator(a)
	cit = newMockIterator(c)
	err = LtVSIter(a, b, a, ait, cit)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(a, expected) {
		t.Errorf("Expected %v, got %v", expected, a)
	}

}
