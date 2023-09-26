package dense

import (
	"github.com/chewxy/inigo/values/tensor"
	"github.com/chewxy/inigo/values/tensor/internal"
	"github.com/chewxy/inigo/values/tensor/internal/flatiter"
	"gorgonia.org/dtype"
)

type AP = tensor.AP
type SliceRange = internal.SliceRange
type DataOrder = internal.DataOrder
type MemoryFlag = internal.MemoryFlag
type ConsOpt = tensor.ConsOpt
type Constructor = tensor.Constructor
type FuncOpt = tensor.FuncOpt

type Engine = internal.Engine
type NonStandardEngine = tensor.NonStandardEngine
type Memory = internal.Memory
type Iterator = internal.Iterator
type Memsetter = tensor.Memsetter
type Desc = tensor.Desc
type DescWithStorage = tensor.DescWithStorage

const AllAxes = internal.AllAxes

func SR(start int, opts ...int) SliceRange { return internal.SR(start, opts...) }

type Option = tensor.Option

func newIterator(t Desc) Iterator {
	return flatiter.New(t.Info())
}

type Num = internal.Num
type OrderedNum = internal.OrderedNum

/*
   Utils
These functions are already defined in internal. A thin wrapper is placed here.

*/

/* ConsOpt */

func WithBacking(backing any) ConsOpt { return tensor.WithBacking(backing) }

func WithShape(shp ...int) ConsOpt { return tensor.WithShape(shp...) }

func WithEngine(e Engine) ConsOpt { return tensor.WithEngine(e) }

func FromScalar(a any) ConsOpt { return tensor.FromScalar(a) }

func FromMemory(mem Memory) ConsOpt { return tensor.FromMemory(mem) }

func AsFortran(backing ...any) ConsOpt { return tensor.AsFortran(backing...) }

/* Func Opts */

func ParseFuncOpts(opts ...FuncOpt) tensor.Option { return tensor.ParseFuncOpts(opts...) }

func UseUnsafe() FuncOpt { return tensor.UseUnsafe }

func WithReuse(a any) FuncOpt { return tensor.WithReuse(a) }

func WithIncr(a any) FuncOpt { return tensor.WithIncr(a) }

func Along(axes ...int) FuncOpt { return tensor.Along(axes...) }

func As(dt dtype.Datatype) FuncOpt { return tensor.As(dt) }

/* utils */
