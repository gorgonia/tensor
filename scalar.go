// +build ignore

package tensor

import (
	"fmt"
	"io"
	"reflect"
	"unsafe"

	"gorgonia.org/dtype"

	"github.com/pkg/errors"
	"gorgonia.org/tensor/internal/storage"
)

var _ Tensor = Scalar{}
var _ ScalarRep = Scalar{}
var _ ScalarRep = ScalarDense{}

// ScalarDense wraps a *Dense to provide a typesafe alternative for a scalar to be represented in a *Dense.
type ScalarDense struct {
	*Dense
}

func (s ScalarDense) IsScalar() bool { return true }

func (s ScalarDense) ScalarValue() interface{} { return s.Dense.Data() }

// Scalar is a representation of a scalar value on the CPU.
type Scalar struct{ v interface{} }

func MakeScalar(v interface{}) Scalar {
	if s, ok := v.(Scalar); ok {
		return s
	}
	if s, ok := v.(*Scalar); ok {
		return Scalar{s.v}
	}
	return Scalar{v}
}

func (s Scalar) Shape() Shape           { return ScalarShape() }
func (s Scalar) Strides() []int         { return nil }
func (s Scalar) Dtype() dtype.Dtype     { return dtype.Dtype{reflect.TypeOf(s.v)} }
func (s Scalar) Dims() int              { return 0 }
func (s Scalar) Size() int              { return 0 } // TODO
func (s Scalar) DataSize() int          { return 0 }
func (s Scalar) RequiresIterator() bool { return false }
func (s Scalar) Iterator() Iterator     { return nil }
func (s Scalar) DataOrder() DataOrder   { return 0 } // TODO

func (s Scalar) Slice(...Slice) (View, error)                          { return nil, errors.New("Cannot slice a scalar") }
func (s Scalar) At(at ...int) (interface{}, error)                     { return nil, errors.New("Get a value of a scalar") }
func (s Scalar) SetAt(_ interface{}, _ ...int) error                   { return errors.New("Cannot set value of scalar") }
func (s Scalar) Reshape(_ ...int) error                                { return errors.New("Cannot reshape a scalar") }
func (s Scalar) T(_ ...int) error                                      { return errors.New("Cannot transpose a scalar") }
func (s Scalar) UT()                                                   {}
func (s Scalar) Transpose() error                                      { return errors.New("Cannot transpose a scalar") }
func (s Scalar) Apply(fn interface{}, opts ...FuncOpt) (Tensor, error) { return nyierr(typeNYI, s) }

func (s Scalar) Zero()                     {} //TODO
func (s Scalar) Memset(interface{}) error  { return errors.New("Cannot Memset") }
func (s Scalar) Data() interface{}         { return s.v }
func (s Scalar) Eq(other interface{}) bool { return s == other }
func (s Scalar) Clone() interface{}        { return s }

func (s Scalar) IsScalar() bool           { return true }
func (s Scalar) ScalarValue() interface{} { return s.v }

func (s Scalar) Engine() Engine             { return nil }
func (s Scalar) MemSize() uintptr           { return 0 }
func (s Scalar) Uintptr() uintptr           { return 0 }
func (s Scalar) Pointer() unsafe.Pointer    { return nil }
func (s Scalar) IsNativelyAccessible() bool { return true }
func (s Scalar) IsManuallyManaged() bool    { return false }

func (s Scalar) Format(t fmt.State, c rune) {} // TODO
func (s Scalar) String() string             { return fmt.Sprintf("%v", s) }

func (s Scalar) WriteNpy(io.Writer) error   { return nyierr(typeNYI, s) }
func (s Scalar) ReadNpy(io.Reader) error    { return nyierr(typeNYI, s) }
func (s Scalar) GobEncode() ([]byte, error) { return nil, nyierr(typeNYI, s) }
func (s Scalar) GobDecode([]byte) error     { return nyierr(typeNYI, s) }

func (s Scalar) standardEngine() StandardEngine { return StdEng{} }
func (s Scalar) hdr() *storage.Header           { return nil }
func (s Scalar) arr() array                     { return array{} }
func (s Scalar) arrPtr() *array                 { return nil }
