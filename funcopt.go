package tensor

import (
	"context"

	"gorgonia.org/dtype"
)

// FuncOpt is an option that modifies the operation of a given function.
type FuncOpt func(options *Option)

type Option struct {
	Ctx       context.Context
	Reuse     any
	Incr      bool
	Unsafe    bool
	Broadcast bool
	AsType    dtype.Dtype
	Along     []int // only used for axial operations

}

func (o Option) Safe() bool { return !o.Unsafe }

func ParseFuncOpts(opts ...FuncOpt) Option {
	retVal := Option{
		Ctx: context.Background(),
	}
	for _, o := range opts {
		o(&retVal)
	}
	return retVal
}

func UseUnsafe(o *Option)         { o.Unsafe = true }
func UseSafe(o *Option)           { o.Unsafe = false }
func WithReuse(reuse any) FuncOpt { return func(o *Option) { o.Reuse = reuse } }
func WithIncr(incr any) FuncOpt   { return func(o *Option) { o.Reuse = incr; o.Incr = true } }
func Along(axes ...int) FuncOpt   { return func(o *Option) { o.Along = axes } }
func As(dt dtype.Dtype) FuncOpt   { return func(o *Option) { o.AsType = dt } }
func AutoBroadcast(o *Option)     { o.Broadcast = true }
