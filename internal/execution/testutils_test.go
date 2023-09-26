package execution

import (
	"github.com/chewxy/inigo/values/tensor/internal/errors"
	"gorgonia.org/shapes"
)

type mockIterator[T any] struct {
	values []T
	valid  []bool
	idx    int
}

func newMockIterator[T any](data []T) *mockIterator[T] {
	valids := make([]bool, len(data))
	for i := range valids {
		valids[i] = true
	}
	return &mockIterator[T]{
		values: data,
		valid:  valids,
	}
}

func (m *mockIterator[T]) NextValidity() (int, bool, error) {
	idx, err := m.Next()
	if err != nil {
		return idx, false, err
	}
	return idx, m.valid[idx], nil
}

func (m *mockIterator[T]) Next() (int, error) {
	if m.idx == len(m.values) {
		return 0, errors.NoOp{}
	}
	idx := m.idx
	m.idx++
	return idx, nil
}

// Start returns the first index
func (*mockIterator[T]) Start() (int, error) {
	panic("not implemented") // TODO: Implement
}

// NextValid returns the next valid index, as well as a skip count.
func (*mockIterator[T]) NextValid() (int, int, error) {
	panic("not implemented") // TODO: Implement
}

// NextInvalid returns the next invalid index, as well as a skip count.
func (*mockIterator[T]) NextInvalid() (int, int, error) {
	panic("not implemented") // TODO: Implement
}

// Reset resets the iterator
func (*mockIterator[T]) Reset() {
	panic("not implemented") // TODO: Implement
}

// SetReverse tells the iterator to iterate in reverse
func (*mockIterator[T]) SetReverse() {
	panic("not implemented") // TODO: Implement
}

// SetForward tells the iterator to iterate forwards
func (*mockIterator[T]) SetForward() {
	panic("not implemented") // TODO: Implement
}

// Coord returns the coordinates
func (*mockIterator[T]) Coord() []int {
	panic("not implemented") // TODO: Implement
}

// Done returns true when the iterator is done iterating.
func (*mockIterator[T]) Done() bool {
	panic("not implemented") // TODO: Implement
}

// Shape returns the shape of the multidimensional tensor it's iterating on.
func (*mockIterator[T]) Shape() shapes.Shape {
	panic("not implemented") // TODO: Implement
}
