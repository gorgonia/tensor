package dense

import (
	"bytes"
	"fmt"
	"reflect"
	"strconv"

	"github.com/chewxy/inigo/values/tensor"
	"github.com/chewxy/inigo/values/tensor/internal"
)

var fmtFlags = [...]rune{'+', '-', '#', ' ', '0'}
var fmtByte = []byte("%")
var precByte = []byte(".")
var newline = []byte("\n")

var (
	matFirstStart = []byte("⎡")
	matFirstEnd   = []byte("⎤\n")
	matLastStart  = []byte("⎣")
	matLastEnd    = []byte("⎦\n")
	rowStart      = []byte("⎢")
	rowEnd        = []byte("⎥\n")
	vecStart      = []byte("[")
	vecEnd        = []byte("]")
	colVecStart   = []byte("C[")
	rowVecStart   = []byte("R[")

	hElisionCompact = []byte("⋯ ")
	hElision        = []byte("... ")
	vElisionCompact = []byte("  ⋮  \n")
	vElision        = []byte(".\n.\n.\n")

	ufVec    = []byte("Vector")
	ufMat    = []byte("Matrix")
	ufTensor = []byte("Tensor-")

	hInvalid = []byte("--")
)

type fmtState[T any] struct {
	fmt.State

	buf                *bytes.Buffer
	pad                []byte
	hElision, vElision []byte

	meta bool
	flat bool
	ext  bool // extended (i.e no elision)
	comp bool // compact
	c    rune // c is here mainly for struct packing reasons

	w, p int // width and precision
	base int // used only for int/byte arrays

	rows, cols int
	pr, pc     int // printed row, printed col
}

func newFmtState[T any](f fmt.State, c rune) *fmtState[T] {
	retVal := &fmtState[T]{
		State: f,
		buf:   bytes.NewBuffer(make([]byte, 10)),
		c:     c,

		meta:     f.Flag('+'),
		flat:     f.Flag('-'),
		ext:      f.Flag('#'),
		comp:     c == 's',
		hElision: hElision,
		vElision: vElision,
	}

	w, _ := f.Width()
	p, _ := f.Precision()
	retVal.w = w
	retVal.p = p
	return retVal
}

func (f *fmtState[T]) originalFmt() string {
	buf := bytes.NewBuffer(fmtByte)
	for _, flag := range fmtFlags {
		if f.Flag(int(flag)) {
			buf.WriteRune(flag)
		}
	}

	// width
	if w, ok := f.Width(); ok {
		buf.WriteString(strconv.Itoa(w))
	}

	// precision
	if p, ok := f.Precision(); ok {
		buf.Write(precByte)
		buf.WriteString(strconv.Itoa(p))
	}

	buf.WriteRune(f.c)
	return buf.String()

}

func (f *fmtState[T]) cleanFmt() string {
	buf := bytes.NewBuffer(fmtByte)

	// width
	if w, ok := f.Width(); ok {
		buf.WriteString(strconv.Itoa(w))
	}

	// precision
	if p, ok := f.Precision(); ok {
		buf.Write(precByte)
		buf.WriteString(strconv.Itoa(p))
	}

	buf.WriteRune(f.c)
	return buf.String()
}

// does the calculation for metadata
func (f *fmtState[T]) populate(t *Dense[T]) {
	switch {
	case t.IsVector():
		f.rows = 1
		f.cols = t.Size()
	case t.IsScalarEquiv():
		f.rows = 1
		f.cols = 1
	default:
		f.rows = t.Shape()[t.Dims()-2]
		f.cols = t.Shape()[t.Dims()-1]
	}

	switch {
	case f.flat && f.ext:
		f.pc = len(t.Data())
	case f.flat && f.comp:
		f.pc = 5
		f.hElision = hElisionCompact
	case f.flat:
		f.pc = 10
	case f.ext:
		f.pc = f.cols
		f.pr = f.rows
	case f.comp:
		f.pc = internal.Min[int](f.cols, 4)
		f.pr = internal.Min[int](f.rows, 4)
		f.hElision = hElisionCompact
		f.vElision = vElisionCompact
	default:
		f.pc = internal.Min[int](f.cols, 8)
		f.pr = internal.Min[int](f.rows, 8)
	}

}

func (f *fmtState[T]) acceptableRune(d *Dense[T]) {
	if f.c == 'H' {
		f.meta = true
		return // accept H as header only
	}
	switch d.t.Kind() {
	case reflect.Float64:
		switch f.c {
		case 'f', 'e', 'E', 'G', 'b':
		default:
			f.c = 'g'
		}
	case reflect.Float32:
		switch f.c {
		case 'f', 'e', 'E', 'G', 'b':
		default:
			f.c = 'g'
		}
	case reflect.Int, reflect.Int64, reflect.Int32, reflect.Int16, reflect.Int8:
		switch f.c {
		case 'b':
			f.base = 2
		case 'd':
			f.base = 10
		case 'o':
			f.base = 8
		case 'x', 'X':
			f.base = 16
		default:
			f.base = 10
			f.c = 'd'
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		switch f.c {
		case 'b':
			f.base = 2
		case 'd':
			f.base = 10
		case 'o':
			f.base = 8
		case 'x', 'X':
			f.base = 16
		default:
			f.base = 10
			f.c = 'd'
		}
	case reflect.Bool:
		f.c = 't'
	default:
		f.c = 'v'
	}
}

func (f *fmtState[T]) calcWidth(d *Dense[T]) {
	format := f.cleanFmt()
	f.w = 0
	data := d.Data()
	for i := 0; i < len(data); i++ {
		w, _ := fmt.Fprintf(f.buf, format, data[i])
		if w > f.w {
			f.w = w
		}
		f.buf.Reset()
	}
}

func (f *fmtState[T]) makePad() {
	f.pad = make([]byte, max[int](f.w, 2))
	for i := range f.pad {
		f.pad[i] = ' '
	}
}

func (f *fmtState[T]) writeHElision() {
	f.Write(f.hElision)
}

func (f *fmtState[T]) writeVElision() {
	f.Write(f.vElision)
}

// Format implements fmt.Formatter. Formatting can be controlled with verbs and flags. All default Go verbs are supported and work as expected.
// By default, only 8 columns and rows are printed (the first and the last 4 columns and rows, while the middle columns and rows are ellided)
// Special flags are:
//
//	'-' for printing a flat array of values
//	'+' for printing extra metadata before printing the tensor (it prints shape, stride and type, which are useful for debugging)
//	'#' for printing the full tensor - there are no elisions. Overrides the 's' verb
//
// Special care also needs be taken for the verb 's' - it prints a super compressed version of the tensor, only printing 4 cols and 4 rows.
func (t *Dense[T]) Format(s fmt.State, c rune) {
	if c == 'i' {
		fmt.Fprintf(s, "INFO:\n\tAP:  %v\n\tENGINE: %T\n", t.AP, t.e)
		return
	}

	f := newFmtState[T](s, c)

	f.acceptableRune(t)
	f.calcWidth(t)
	f.makePad()
	f.populate(t)

	if f.meta {
		switch {
		case t.IsVector():
			f.Write(ufVec)
		case t.Dims() == 2:
			f.Write(ufMat)
		default:
			f.Write(ufTensor)
			fmt.Fprintf(f, "%d", t.Dims())
		}
		fmt.Fprintf(f, " %v %v\n", t.Shape(), t.Strides())
	}

	if f.c == 'H' {
		return
	}

	if !t.IsNativelyAccessible() {
		fmt.Fprintf(f, "Inaccesible data")
		return
	}

	format := f.cleanFmt()

	if f.flat {
		f.Write(vecStart)
		data := t.Data()
		switch {
		case f.ext:
			for i := 0; i < len(data); i++ {
				// if t.mask[i] {
				// 	fmt.Fprintf(f, "%s", hInvalid)
				// } else {
				// 	fmt.Fprintf(f, format, t.Get(i))
				// }
				fmt.Fprintf(f, format, data[i])

				if i < len(data)-1 {
					f.Write(f.pad[:1])
				}
			}
		// case t.viewOf != 0:
		// 	it := IteratorFromDense(t)
		// 	var c, i int
		// 	var err error
		// 	for i, err = it.Next(); err == nil; i, err = it.Next() {

		// 		if t.mask[i] {
		// 			fmt.Fprintf(f, "%s", hInvalid)
		// 		} else {
		// 			fmt.Fprintf(f, format, t.Get(i))
		// 		}

		// 		f.Write(f.pad[:1])

		// 		c++
		// 		if c >= f.pc {
		// 			f.writeHElision()
		// 			break
		// 		}
		// 	}
		// 	if err != nil {
		// 		if _, noop := err.(NoOpError); !noop {
		// 			fmt.Fprintf(f, "ERROR ITERATING: %v", err)

		// 		}
		// 	}
		default:

			for i := 0; i < f.pc; i++ {
				fmt.Fprintf(f, format, data[i])
				// if !t.IsMasked() {
				// 	fmt.Fprintf(f, format, t.Get(i))
				// } else {
				// 	if t.mask[i] {
				// 		fmt.Fprintf(f, "%s", hInvalid)
				// 	} else {
				// 		fmt.Fprintf(f, format, t.Get(i))
				// 	}
				// }
				f.Write(f.pad[:1])
			}
			if f.pc < len(data) {
				f.writeHElision()
			}
		}
		f.Write(vecEnd)
		return
	}

	// standard stuff
	it := newIterator(t)
	coord := it.Coord()

	firstRow := true
	firstVal := true
	var lastRow, lastCol int
	var expected int
	data := t.Data()
	for next, err := it.Next(); err == nil; next, err = it.Next() {
		if next < expected {
			continue
		}

		var col, row int
		row = lastRow
		col = lastCol
		if f.rows > f.pr && row > f.pr/2 && row < f.rows-f.pr/2 {
			continue
		}

		if firstVal {
			if firstRow {
				switch {
				case t.IsColVec():
					f.Write(colVecStart)
				case t.IsRowVec():
					f.Write(rowVecStart)
				case t.IsVector():
					f.Write(vecStart)
				case t.IsScalarEquiv():
					for i := 0; i < t.Dims(); i++ {
						f.Write(vecStart)
					}
				default:
					f.Write(matFirstStart)
				}

			} else {
				var matLastRow bool
				if !t.IsVector() {
					matLastRow = coord[len(coord)-2] == f.rows-1
				}
				if matLastRow {
					f.Write(matLastStart)
				} else {
					f.Write(rowStart)
				}
			}
			firstVal = false
		}

		// actual printing of the value
		if f.cols <= f.pc || (col < f.pc/2 || (col >= f.cols-f.pc/2)) {
			var w int
			w, _ = fmt.Fprintf(f.buf, format, data[next])
			// f t.IsMasked() {
			// 	if t.mask[next] {
			// 		w, _ = fmt.Fprintf(f.buf, "%s", hInvalid)
			// 	} else {
			// 		w, _ = fmt.Fprintf(f.buf, format, t.Get(next))
			// 	}
			// } else {
			// 	w, _ = fmt.Fprintf(f.buf, format, t.Get(next))
			// }
			f.Write(f.pad[:f.w-w]) // prepad
			f.Write(f.buf.Bytes()) // write

			if col < f.cols-1 { // pad with a space
				f.Write(f.pad[:2])
			}
			f.buf.Reset()
		} else if col == f.pc/2 {
			f.writeHElision()
		}

		// done printing
		// check for end of rows
		if col == f.cols-1 {
			eom := row == f.rows-1
			switch {
			case t.IsVector():
				f.Write(vecEnd)
				return
			case t.IsScalarEquiv():
				for i := 0; i < t.Dims(); i++ {
					f.Write(vecEnd)
				}
				return
			case firstRow:
				f.Write(matFirstEnd)
			case eom:
				f.Write(matLastEnd)
				if t.IsMatrix() {
					return
				}

				// one newline for every dimension above 2
				for i := t.Dims(); i > 2; i-- {
					f.Write(newline)
				}

			default:
				f.Write(rowEnd)
			}

			if firstRow {
				firstRow = false
			}

			if eom {
				firstRow = true
			}

			firstVal = true

			// figure out elision
			if f.rows > f.pr && row+1 == f.pr/2 {
				expectedCoord := make([]int, len(coord))
				copy(expectedCoord, coord)
				expectedCoord[len(expectedCoord)-2] = f.rows - (f.pr / 2)
				expected, _ = tensor.Ltoi(t.Shape(), t.Strides(), expectedCoord...)

				f.writeVElision()
			}
		}

		// cleanup
		switch {
		case t.IsRowVec():
			lastRow = coord[len(coord)-2]
			lastCol = coord[len(coord)-1]
		case t.IsColVec():
			lastRow = coord[len(coord)-1]
			lastCol = coord[len(coord)-2]
		case t.IsVector():
			lastCol = coord[len(coord)-1]
		default:
			lastRow = coord[len(coord)-2]
			lastCol = coord[len(coord)-1]
		}
	}
}
