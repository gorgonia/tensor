package dense

import (
	"sync"

	"gorgonia.org/dtype"
)

type ConsFunc func(opts ...ConsOpt) (DescWithStorage, error)

var consRegistryLock sync.Mutex
var consRegistry = map[dtype.Datatype]ConsFunc{
	dtype.Bool:       func(opts ...ConsOpt) (DescWithStorage, error) { return New[bool](opts...), nil },
	dtype.Int:        func(opts ...ConsOpt) (DescWithStorage, error) { return New[int](opts...), nil },
	dtype.Int8:       func(opts ...ConsOpt) (DescWithStorage, error) { return New[int8](opts...), nil },
	dtype.Int16:      func(opts ...ConsOpt) (DescWithStorage, error) { return New[int16](opts...), nil },
	dtype.Int32:      func(opts ...ConsOpt) (DescWithStorage, error) { return New[int32](opts...), nil },
	dtype.Int64:      func(opts ...ConsOpt) (DescWithStorage, error) { return New[int64](opts...), nil },
	dtype.Uint:       func(opts ...ConsOpt) (DescWithStorage, error) { return New[uint](opts...), nil },
	dtype.Uint8:      func(opts ...ConsOpt) (DescWithStorage, error) { return New[uint8](opts...), nil },
	dtype.Uint16:     func(opts ...ConsOpt) (DescWithStorage, error) { return New[uint16](opts...), nil },
	dtype.Uint32:     func(opts ...ConsOpt) (DescWithStorage, error) { return New[uint32](opts...), nil },
	dtype.Uint64:     func(opts ...ConsOpt) (DescWithStorage, error) { return New[uint64](opts...), nil },
	dtype.Float32:    func(opts ...ConsOpt) (DescWithStorage, error) { return New[float32](opts...), nil },
	dtype.Float64:    func(opts ...ConsOpt) (DescWithStorage, error) { return New[float64](opts...), nil },
	dtype.Complex64:  func(opts ...ConsOpt) (DescWithStorage, error) { return New[complex64](opts...), nil },
	dtype.Complex128: func(opts ...ConsOpt) (DescWithStorage, error) { return New[complex128](opts...), nil },
	dtype.String:     func(opts ...ConsOpt) (DescWithStorage, error) { return New[string](opts...), nil },
}

func RegisterDtype(dt dtype.Datatype, fn ConsFunc) {
	consRegistryLock.Lock()
	defer consRegistryLock.Unlock()

	consRegistry[dt] = fn
}
