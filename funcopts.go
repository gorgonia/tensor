package tensor

import (
	"context"

	"gorgonia.org/dtype"
)

// FuncOpt are optionals for calling Tensor functions.
// The `*opOpt` type is unexported, but it's methods are exported.
// This is intentional as use of the `*opOpt` is very specialized.
// See funcopts.go for more information.
type FuncOpt func(*opOpt)

// WithIncr passes in a Tensor to be incremented.
func WithIncr(incr Tensor) FuncOpt {
	f := func(opt *opOpt) {
		opt.incr = incr
	}
	return f
}

// WithReuse passes in a Tensor to be reused.
func WithReuse(reuse Tensor) FuncOpt {
	f := func(opt *opOpt) {
		opt.reuse = reuse
	}
	return f
}

// UseSafe ensures that the operation is a safe operation (copies data, does not clobber). This is the default option for most methods and functions
func UseSafe() FuncOpt {
	f := func(opt *opOpt) {
		opt.unsafe = false
	}
	return f
}

// UseUnsafe ensures that the operation is an unsafe operation - data will be clobbered, and operations performed inplace
func UseUnsafe() FuncOpt {
	f := func(opt *opOpt) {
		opt.unsafe = true
	}
	return f
}

// AsSameType makes sure that the return Tensor is the same type as input Tensors.
func AsSameType() FuncOpt {
	f := func(opt *opOpt) {
		opt.same = true
	}
	return f
}

// As makes sure that the the return Tensor is of the type specified. Currently only works for FromMat64
func As(t dtype.Dtype) FuncOpt {
	f := func(opt *opOpt) {
		opt.t = t
	}
	return f
}

// WithContext allows a function to be called with a given context
func WithContext(ctx context.Context) FuncOpt {
	f := func(opt *opOpt) {
		opt.ctx = ctx
	}
	return f
}

// opOpt are the options used to call ops
type opOpt struct {
	reuse  Tensor
	incr   Tensor
	unsafe bool
	same   bool
	t      dtype.Dtype
	ctx    context.Context
}

// ParseFuncOpts parses a list of FuncOpt into a single unified method call structure.
func ParseFuncOpts(opts ...FuncOpt) *opOpt {
	retVal := borrowOpOpt()

	for _, opt := range opts {
		opt(retVal)
	}
	if retVal.ctx == nil {
		retVal.ctx = context.Background() // default context - required for no panics.
	}
	return retVal
}

// Incr returns the tensor to be incremented in the call. Can be nil.
func (fo *opOpt) Incr() Tensor { return fo.incr }

// Reuse returns the tensor to be reused in the call. Can be nil.
func (fo *opOpt) Reuse() Tensor { return fo.reuse }

// IncrReuse returns whether a reuse tensor is to be used as the incr Tensor
func (fo *opOpt) IncrReuse() (Tensor, bool) {
	if fo.incr != nil {
		return fo.incr, true
	}
	return fo.reuse, false
}

// Safe signals if the op is to be done safely
func (fo *opOpt) Safe() bool { return !fo.unsafe }

// Same signals if the op is to return the same type as its inputs
func (fo *opOpt) Same() bool { return fo.same }

// As returns the dtype of the return value of the method call.
// For example:
//		a.Lt(b, As(Bool))
// indicates that the result of the `Lt()` should be a Tensor of Bool.
//
// Another example:
//		a.Add(b, As(Int))
// indicates that the result of `Add()` should be converted to a Tensor of Int.
// Note that this function is not yet supported in most operations.
func (fo *opOpt) As() dtype.Dtype { return fo.t }

// Context returns a context.Context that may have been passed in as a function option.
func (fo *opOpt) Context() context.Context { return fo.ctx }

// SetReuse allows the reuse parameter to be set.
func (fo *opOpt) SetReuse(reuse Tensor) { fo.reuse = reuse }

// SetIncr allows the incr parameter to be set.
func (fo *opOpt) SetIncr(incr Tensor) { fo.incr = incr }

// FuncOpts is the inverse of ParseFuncOpts.
func (fo *opOpt) FuncOpts() []FuncOpt {
	retVal := make([]FuncOpt, 0, 4)
	if fo.reuse != nil {
		retVal = append(retVal, WithReuse(fo.reuse))
	}
	if fo.incr != nil {
		retVal = append(retVal, WithIncr(fo.incr))
	}
	if fo.unsafe {
		retVal = append(retVal, UseUnsafe())
	}
	if fo.same {
		retVal = append(retVal, AsSameType())
	}
	if fo.t != (Dtype{}) {
		retVal = append(retVal, As(fo.t))
	}
	return retVal
}
