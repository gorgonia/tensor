package paginated

import "errors"

var (
	ErrCache          = errors.New("cache error")
	ErrFilled         = errors.New("tensor capacity is filled")
	ErrLength         = errors.New("incorrect length")
	ErrType           = errors.New("invalid type assertion")
	ErrFuncType       = errors.New("invalid function type assertion")
	ErrDNE            = errors.New("does not exist")
	ErrBound          = errors.New("exceeds bound")
	ErrDims           = errors.New("incorrect dimensions")
	ErrSize           = errors.New("incorrect size")
	ErrIndex          = errors.New("index out of bounds")
	ErrFormat         = errors.New("unsupported format")
	ErrNotImplemented = errors.New("not implemented")
)
