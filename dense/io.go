package dense

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"

	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/serialization/pb"

	"google.golang.org/protobuf/proto"
)

/* GOB ENCODING / DECODING */

// GobEncode implements the gob.GobEncoder interface.
func (t *Dense[DT]) GobEncode() (p []byte, err error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	if err = encoder.Encode(t.AP); err != nil {
		return
	}

	if err = encoder.Encode(t.Array); err != nil {
		return
	}

	return buf.Bytes(), err
}

// GobDecode implements the gob.GobDecoder interface.
func (t *Dense[DT]) GobDecode(p []byte) (err error) {
	buf := bytes.NewBuffer(p)
	decoder := gob.NewDecoder(buf)

	if err = decoder.Decode(&t.AP); err != nil {
		return
	}

	if err = decoder.Decode(&t.Array); err != nil {
		return
	}

	return nil

}

/* NUMPY ENCODING/DECODING */

var npyDescRE = regexp.MustCompile(`'descr':\s*'([^']*)'`)
var rowOrderRE = regexp.MustCompile(`'fortran_order':\s*(False|True)`)
var shapeRE = regexp.MustCompile(`'shape':\s*\(([^\(]*)\)`)

type binaryWriter struct {
	io.Writer
	err error
	seq int
}

func (w *binaryWriter) w(x interface{}) {
	if w.err != nil {
		return
	}

	w.err = binary.Write(w, binary.LittleEndian, x)
	w.seq++
}

func (w *binaryWriter) Err() error {
	if w.err == nil {
		return nil
	}
	return errors.Wrapf(w.err, "Sequence %d", w.seq)
}

type binaryReader struct {
	io.Reader
	err error
	seq int
}

func (r *binaryReader) Read(data interface{}) {
	if r.err != nil {
		return
	}
	r.err = binary.Read(r.Reader, binary.LittleEndian, data)
	r.seq++
}

func (r *binaryReader) Err() error {
	if r.err == nil {
		return nil
	}
	return errors.Wrapf(r.err, "Sequence %d", r.seq)
}

// WriteNpy writes the *Tensor as a numpy compatible serialized file.
//
// The format is very well documented here:
// https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
//
// Gorgonia specifically uses Version 1.0, as 65535 bytes should be more than enough for the headers.
// The values are written in little endian order, because let's face it -
// 90% of the world's computers are running on x86+ processors.
//
// This method does not close the writer. Closing (if needed) is deferred to the caller
// If tensor is masked, invalid values are replaced by the default fill value.
func (t *Dense[DT]) WriteNpy(w io.Writer) (err error) {
	var npdt string
	if npdt, err = t.t.NumpyDtype(); err != nil {
		return
	}

	var header string
	if t.Dims() == 1 {
		// when t is a 1D vector, numpy expects "(N,)" instead of "(N)" which t.Shape() returns.
		header = "{'descr': '<%v', 'fortran_order': False, 'shape': (%d,), }"
		header = fmt.Sprintf(header, npdt, t.Shape()[0])
	} else {
		header = "{'descr': '<%v', 'fortran_order': False, 'shape': %v, }"
		header = fmt.Sprintf(header, npdt, t.Shape())
	}
	padding := 64 - ((10 + len(header)) % 64) // old version used 16. As of np 1.2, a requirement is that it is evenly divisible by 64 for alignment purposes
	if padding > 0 {
		header = header + strings.Repeat(" ", padding-1) + "\n" // for binary compatibility as at numpy 1.21.5, the last paddding is replaced with a LF. binary compatibility is only for tests (see TestDense_Numpy)
	}
	bw := binaryWriter{Writer: w}
	bw.Write([]byte("\x93NUMPY")) // stupid magic
	bw.w(byte(1))                 // major version
	bw.w(byte(0))                 // minor version
	bw.w(uint16(len(header)))     // 4 bytes to denote header length
	if err = bw.Err(); err != nil {
		return err
	}
	bw.Write([]byte(header))

	bw.seq = 0

	// special handling of int and uint
	var v DT
	switch any(v).(type) {
	case int:
		data := any(t.Data()).([]int)
		for _, v := range data {
			bw.w(int64(v))
		}
	case uint:
		data := any(t.Data()).([]uint)
		for _, v := range data {
			bw.w(uint64(v))
		}
	default:
		for _, v := range t.Data() {
			bw.w(v)
		}
	}

	return bw.Err()
}

func FromNpy(r io.Reader) (retVal DescWithStorage, err error) {
	br := binaryReader{Reader: r}

	var magic [6]byte
	if br.Read(magic[:]); string(magic[:]) != "\x93NUMPY" {
		return nil, errors.Errorf("Not a numpy file. Got %q as the magic number instead", string(magic[:]))
	}

	var version, minor byte
	if br.Read(&version); version != 1 {
		return nil, errors.New("Only verion 1.0 of numpy's serialization format is currently supported (65535 bytes ought to be enough for a header)")
	}

	if br.Read(&minor); minor != 0 {
		return nil, errors.New("Only verion 1.0 of numpy's serialization format is currently supported (65535 bytes ought to be enough for a header)")
	}

	var headerLen uint16
	br.Read(&headerLen)
	header := make([]byte, int(headerLen))
	br.Read(header)
	if err = br.Err(); err != nil {
		return
	}

	// extract stuff from header
	var match [][]byte
	if match = npyDescRE.FindSubmatch(header); match == nil {
		return nil, errors.New("No dtype information in npy file")
	}

	// TODO: check for endianness. For now we assume everything is little endian
	var dt dtype.Dtype
	if dt, err = dtype.FromNumpyDtype(string(match[1][1:])); err != nil {
		return
	}

	if match = rowOrderRE.FindSubmatch(header); match == nil {
		return nil, errors.New("No Row Order information found in the numpy file")
	}
	if string(match[1]) != "False" {
		return nil, errors.New("Cannot yet read from Fortran Ordered Numpy files")
	}

	if match = shapeRE.FindSubmatch(header); match == nil {
		return nil, errors.New("No shape information found in npy file")
	}
	sizesStr := strings.Split(string(match[1]), ",")

	var shape shapes.Shape
	for _, s := range sizesStr {
		s = strings.Trim(s, " ")
		if len(s) == 0 {
			break
		}
		var size int
		if size, err = strconv.Atoi(s); err != nil {
			return
		}
		shape = append(shape, size)
	}

	// make data then fill with the read values
	data := dt.SliceOf(shape.TotalSize())

	// read values into `data` slice.
	// special handling of int and uint is handled in package dtype
	dt.ReadIntoSlice(data, &br)

	// now create a *Dense[??]

	return NewOf(dt, WithBacking(data), WithShape(shape...))
}

/* PROTOBUF  ENCODING/DECODING */

func (t *Dense[DT]) ToPB() *pb.Dense {
	retVal := new(pb.Dense)
	retVal.Ap = t.AP.ToPB()
	retVal.Data = t.Array.DataAsBytes()
	retVal.TransposedWith = make([]int32, len(t.transposedWith))
	for i := range retVal.TransposedWith {
		retVal.TransposedWith[i] = int32(t.transposedWith[i])
	}
	retVal.Memoryflag = uint32(t.f)
	retVal.Type = t.t.Name()
	return retVal
}

func (t *Dense[DT]) PBEncode() ([]byte, error) {
	pbDense := t.ToPB()
	return proto.Marshal(pbDense)
}
