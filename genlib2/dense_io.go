package main

import (
	"fmt"
	"io"
	"text/template"
)

const writeNpyRaw = `
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
// http://docs.scipy.org/doc/numpy/neps/npy-format.html
//
// Gorgonia specifically uses Version 1.0, as 65535 bytes should be more than enough for the headers.
// The values are written in little endian order, because let's face it -
// 90% of the world's computers are running on x86+ processors.
//
// This method does not close the writer. Closing (if needed) is deferred to the caller
// If tensor is masked, invalid values are replaced by the default fill value.
func (t *Dense) WriteNpy(w io.Writer) (err error) {
	var npdt string
	if npdt, err = t.t.numpyDtype(); err != nil{
		return
	}

	var header string
	if t.Dims() == 1 {
		// when t is a 1D vector, numpy expects "(N,)" instead of "(N)" which t.Shape() returns.
		header = "{'descr': '<%v', 'fortran_order': False, 'shape': (%d,)}"
		header = fmt.Sprintf(header, npdt, t.Shape()[0])
	} else {
		header = "{'descr': '<%v', 'fortran_order': False, 'shape': %v}"
		header = fmt.Sprintf(header, npdt, t.Shape())
	}
	padding := 16 - ((10 + len(header)) % 16)
	if padding > 0 {
		header = header + strings.Repeat(" ", padding)
	}
	bw := binaryWriter{Writer: w}
	bw.Write([]byte("\x93NUMPY"))                              // stupid magic
	bw.w(byte(1))             // major version
	bw.w(byte(0))             // minor version
	bw.w(uint16(len(header))) // 4 bytes to denote header length
	if err = bw.Err() ; err != nil {
		return err
	}
	bw.Write([]byte(header))

	bw.seq = 0
	if t.IsMasked(){
		fillval:=t.FillValue()
		it := FlatMaskedIteratorFromDense(t)
		for i, err := it.Next(); err == nil; i, err = it.Next() {
			if t.mask[i] {
				bw.w(fillval)
			} else{
				bw.w(t.Get(i))
			}
		}
	} else {
		for i := 0; i < t.len(); i++ {
			bw.w(t.Get(i))
		}
	}

		return bw.Err()
}
`

const writeCSVRaw = `// WriteCSV writes the *Dense to a CSV. It accepts an optional string formatting ("%v", "%f", etc...), which controls what is written to the CSV.
// If tensor is masked, invalid values are replaced by the default fill value.
func (t *Dense) WriteCSV(w io.Writer, formats ...string) (err error) {
	// checks:
	if !t.IsMatrix() {
		// error
		err = errors.Errorf("Cannot write *Dense to CSV. Expected number of dimensions: <=2, T has got %d dimensions (Shape: %v)", t.Dims(), t.Shape())
		return
	}
	format := "%v"
	if len(formats) > 0{
		format = formats[0]
	}

	cw := csv.NewWriter(w)
	it := IteratorFromDense(t)
	coord := it.Coord()

	// rows := t.Shape()[0]
	cols := t.Shape()[1]
	record := make([]string, 0, cols)
	var i, k, lastCol int
	isMasked:=t.IsMasked()
	fillval:= t.FillValue()
	fillstr:= fmt.Sprintf(format, fillval)
	for i, err = it.Next(); err == nil; i, err = it.Next() {
		record = append(record, fmt.Sprintf(format, t.Get(i)))
		if isMasked{
			if t.mask[i] {
				record[k]=fillstr
				}
				k++
		}
		if lastCol == cols-1 {
			if err = cw.Write(record); err != nil {
				// TODO: wrap errors
				return
			}
			cw.Flush()
			record = record[:0]
		}

		// cleanup
		switch {
		case t.IsRowVec():
			// lastRow = coord[len(coord)-2]
			lastCol = coord[len(coord)-1]
		case t.IsColVec():
			// lastRow = coord[len(coord)-1]
			lastCol = coord[len(coord)-2]
		case t.IsVector():
			lastCol = coord[len(coord)-1]
		default:
			// lastRow = coord[len(coord)-2]
			lastCol = coord[len(coord)-1]
		}
	}
	return nil
}

`

const gobEncodeRaw = `// GobEncode implements gob.GobEncoder
func (t *Dense) GobEncode() (p []byte, err error){
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	if err = encoder.Encode(t.Shape()); err != nil {
		return
	}

	if err = encoder.Encode(t.Strides()); err != nil {
		return
	}

	if err = encoder.Encode(t.AP.o); err != nil {
		return
	}

	if err = encoder.Encode(t.AP.Δ); err != nil {
		return
	}

	if err = encoder.Encode(t.mask); err != nil {
		return
	}

	data := t.Data()
	if err = encoder.Encode(&data); err != nil {
		return
	}

	return buf.Bytes(), err
}
`

const gobDecodeRaw = `// GobDecode implements gob.GobDecoder
func (t *Dense) GobDecode(p []byte) (err error){
	buf := bytes.NewBuffer(p)
	decoder := gob.NewDecoder(buf)


	var shape Shape
	if err = decoder.Decode(&shape); err != nil {
		return
	}

	var strides []int
	if err = decoder.Decode(&strides); err != nil {
		return
	}

	var o DataOrder
	var tr Triangle
	if err = decoder.Decode(&o); err == nil {
		if err = decoder.Decode(&tr); err != nil {
			return
		}
	}

	t.AP.Init(shape, strides)
	t.AP.o = o
	t.AP.Δ = tr

	var mask []bool
	if err = decoder.Decode(&mask); err != nil {
		return
	}

	var data interface{}
	if err = decoder.Decode(&data); err != nil {
		return
	}

	t.fromSlice(data)
	t.addMask(mask)
	t.fix()
	if t.e == nil {
		t.e = StdEng{}
	}
	return t.sanity()
}
`
const npyDescRE = `var npyDescRE = regexp.MustCompile(` + "`" + `'descr':` + `\` + `s*'([^']*)'` + "`" + ")"
const rowOrderRE = `var rowOrderRE = regexp.MustCompile(` + "`" + `'fortran_order':\s*(False|True)` + "`)"
const shapeRE = `var shapeRE = regexp.MustCompile(` + "`" + `'shape':\s*\(([^\(]*)\)` + "`)"

const readNpyRaw = `// ReadNpy reads NumPy formatted files into a *Dense
func (t *Dense) ReadNpy(r io.Reader) (err error){
	br := binaryReader{Reader: r}
	var magic [6]byte
	if br.Read(magic[:]); string(magic[:]) != "\x93NUMPY" {
		return errors.Errorf("Not a numpy file. Got %q as the magic number instead", string(magic[:]))
	}

	var version, minor byte
	if br.Read(&version); version != 1 {
		return errors.New("Only verion 1.0 of numpy's serialization format is currently supported (65535 bytes ought to be enough for a header)")
	}

	if br.Read(&minor); minor != 0 {
		return errors.New("Only verion 1.0 of numpy's serialization format is currently supported (65535 bytes ought to be enough for a header)")
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
		return errors.New("No dtype information in npy file")
	}

	// TODO: check for endianness. For now we assume everything is little endian
	if t.t, err = fromNumpyDtype(string(match[1][1:])); err != nil {
		return
	}

	if match = rowOrderRE.FindSubmatch(header); match == nil {
		return errors.New("No Row Order information found in the numpy file")
	}
	if string(match[1]) != "False" {
		return errors.New("Cannot yet read from Fortran Ordered Numpy files")
	}

	if match = shapeRE.FindSubmatch(header); match == nil {
		return  errors.New("No shape information found in npy file")
	}
	sizesStr := strings.Split(string(match[1]), ",")


	var shape Shape
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

	size := shape.TotalSize()
	if t.e == nil {
		t.e = StdEng{}
	}
	t.makeArray(size)

	switch t.t.Kind() {
	{{range .Kinds -}}
	case reflect.{{reflectKind .}}:
		data := t.{{sliceOf .}}
		for i := 0; i < size; i++ {
			br.Read(&data[i])
		}
	{{end -}}
	}
	if err = br.Err(); err != nil {
		return err
	}

	t.AP.zeroWithDims(len(shape))
	t.setShape(shape...)
	t.fix()
	return t.sanity()
}
`

const readCSVRaw = `// convFromStrs converts a []string to a slice of the Dtype provided. It takes a provided backing slice.
// If into is nil, then a backing slice will be created.
func convFromStrs(to Dtype, record []string, into interface{}) (interface{}, error) {
	var err error
	switch to.Kind() {
		{{range .Kinds -}}
		{{if isNumber . -}}
		{{if isOrd . -}}
	case reflect.{{reflectKind .}}:
		retVal := make([]{{asType .}}, len(record))
		var backing []{{asType .}}
		if into == nil {
			backing = make([]{{asType .}}, 0, len(record))
		}else{
			backing = into.([]{{asType .}})
		}

		for i, v := range record {
			{{if eq .String "float64" -}}
				if retVal[i], err = strconv.ParseFloat(v, 64); err != nil {
					return nil, err
				}
			{{else if eq .String "float32" -}}
				var f float64
				if f, err = strconv.ParseFloat(v, 32); err != nil {
					return nil, err
				}
				retVal[i] = float32(f)
			{{else if hasPrefix .String "int" -}}
				var i64 int64
				if i64, err = strconv.ParseInt(v, 10, {{bitSizeOf .}}); err != nil {
					return nil, err
				}
				retVal[i] = {{asType .}}(i64)
			{{else if hasPrefix .String "uint" -}}
				var u uint64
				if u, err = strconv.ParseUint(v, 10, {{bitSizeOf .}}); err != nil {
					return nil, err
				}
				retVal[i] = {{asType .}}(u)
			{{end -}}
		}
		backing = append(backing, retVal...)
		return backing, nil
		{{end -}}
		{{end -}}
		{{end -}}
	case reflect.String:
		var backing []string
		if into == nil {
			backing = make([]string, 0, len(record))
		}else{
			backing = into.([]string)
		}
		backing = append(backing, record...)
		return backing, nil
	default:
		return nil,errors.Errorf(methodNYI, "convFromStrs", to)
	}
}

// ReadCSV reads a CSV into a *Dense. It will override the underlying data.
//
// BUG(chewxy): reading CSV doesn't handle CSVs with different columns per row yet.
func (t *Dense) ReadCSV(r io.Reader, opts ...FuncOpt) (err error) {
	fo := ParseFuncOpts(opts...)
	as := fo.As()
	if as.Type == nil {
		as = Float64
	}

	cr := csv.NewReader(r)

	var record []string
	var rows, cols int
	var backing interface{}
	for {
		record, err = cr.Read()
		if err == io.EOF{
			break
		} else 	if err != nil {
			return
		}
		if backing, err = convFromStrs(as, record, backing); err != nil {
			return
		}
		cols = len(record)
		rows++
	}
	t.fromSlice(backing)
	t.AP.zero()
	t.AP.SetShape(rows, cols)
	return nil
	return errors.Errorf("not yet handled")
}
`

var fbEncodeDecodeRaw = `// FBEncode encodes to a byte slice using flatbuffers.
//
// Only natively accessible data can be encided
func (t *Dense) FBEncode() ([]byte, error) {
	builder := flatbuffers.NewBuilder(1024)

	fb.DenseStartShapeVector(builder, len(t.shape))
	for i := len(t.shape) - 1; i >= 0; i-- {
		builder.PrependInt32(int32(t.shape[i]))
	}
	shape := builder.EndVector(len(t.shape))

	fb.DenseStartStridesVector(builder, len(t.strides))
	for i := len(t.strides) - 1; i >= 0; i-- {
		builder.PrependInt32(int32(t.strides[i]))
	}
	strides := builder.EndVector(len(t.strides))

	var o uint32
	switch {
	case t.o.IsRowMajor() && t.o.IsContiguous():
		o = 0
	case t.o.IsRowMajor() && !t.o.IsContiguous():
		o = 1
	case t.o.IsColMajor() && t.o.IsContiguous():
		o = 2
	case t.o.IsColMajor() && !t.o.IsContiguous():
		o = 3
	}

	var triangle int32
	switch t.Δ {
	case NotTriangle:
		triangle = fb.TriangleNOT_TRIANGLE
	case Upper:
		triangle = fb.TriangleUPPER
	case Lower:
		triangle = fb.TriangleLOWER
	case Symmetric:
		triangle = fb.TriangleSYMMETRIC
	}

	dt := builder.CreateString(t.Dtype().String())
	data := t.byteSlice()

	fb.DenseStartDataVector(builder, len(data))
	for i := len(data) - 1; i >= 0; i-- {
		builder.PrependUint8(data[i])
	}
	databyte := builder.EndVector(len(data))

	fb.DenseStart(builder)
	fb.DenseAddShape(builder, shape)
	fb.DenseAddStrides(builder, strides)
	fb.DenseAddO(builder, o)
	fb.DenseAddT(builder, triangle)
	fb.DenseAddType(builder, dt)
	fb.DenseAddData(builder, databyte)
	serialized := fb.DenseEnd(builder)
	builder.Finish(serialized)

	return builder.FinishedBytes(), nil
}

// FBDecode decodes a byteslice from a flatbuffer table into a *Dense
func (t *Dense) FBDecode(buf []byte) error {
	serialized := fb.GetRootAsDense(buf, 0)

	o := serialized.O()
	switch o {
	case 0:
		t.o = 0
	case 1:
		t.o = MakeDataOrder(NonContiguous)
	case 2:
		t.o = MakeDataOrder(ColMajor)
	case 3:
		t.o = MakeDataOrder(ColMajor, NonContiguous)
	}

	tri := serialized.T()
	switch tri {
	case fb.TriangleNOT_TRIANGLE:
		t.Δ = NotTriangle
	case fb.TriangleUPPER:
		t.Δ = Upper
	case fb.TriangleLOWER:
		t.Δ = Lower
	case fb.TriangleSYMMETRIC:
		t.Δ = Symmetric
	}

	t.shape = Shape(BorrowInts(serialized.ShapeLength()))
	for i := 0; i < serialized.ShapeLength(); i++ {
		t.shape[i] = int(int32(serialized.Shape(i)))
	}

	t.strides = BorrowInts(serialized.StridesLength())
	for i := 0; i < serialized.ShapeLength(); i++ {
		t.strides[i] = int(serialized.Strides(i))
	}
	typ := string(serialized.Type())
	for _, dt := range allTypes.set {
		if dt.String() == typ {
			t.t = dt
			break
		}
	}

	if t.e == nil {
		t.e = StdEng{}
	}
	t.makeArray(t.shape.TotalSize())

	// allocated data. Now time to actually copy over the data
	db := t.byteSlice()
	copy(db, serialized.DataBytes())
	t.fix()
	return t.sanity()
}
`

var pbEncodeDecodeRaw = `// PBEncode encodes the Dense into a protobuf byte slice.
func (t *Dense) PBEncode() ([]byte, error) {
	var toSerialize pb.Dense
	toSerialize.Shape = make([]int32, len(t.shape))
	for i, v := range t.shape {
		toSerialize.Shape[i] = int32(v)
	}
	toSerialize.Strides = make([]int32, len(t.strides))
	for i, v := range t.strides {
		toSerialize.Strides[i] = int32(v)
	}

	switch {
	case t.o.IsRowMajor() && t.o.IsContiguous():
		toSerialize.O = pb.RowMajorContiguous
	case t.o.IsRowMajor() && !t.o.IsContiguous():
		toSerialize.O = pb.RowMajorNonContiguous
	case t.o.IsColMajor() && t.o.IsContiguous():
		toSerialize.O = pb.ColMajorContiguous
	case t.o.IsColMajor() && !t.o.IsContiguous():
		toSerialize.O = pb.ColMajorNonContiguous
	}
	toSerialize.T = pb.Triangle(t.Δ)
	toSerialize.Type = t.t.String()
	data := t.byteSlice()
	toSerialize.Data = make([]byte, len(data))
	copy(toSerialize.Data, data)
	return toSerialize.Marshal()
}

// PBDecode unmarshalls a protobuf byteslice into a *Dense.
func (t *Dense) PBDecode(buf []byte) error {
	var toSerialize pb.Dense
	if err := toSerialize.Unmarshal(buf); err != nil {
		return err
	}
	t.shape = make(Shape, len(toSerialize.Shape))
	for i, v := range toSerialize.Shape {
		t.shape[i] = int(v)
	}
	t.strides = make([]int, len(toSerialize.Strides))
	for i, v := range toSerialize.Strides {
		t.strides[i] = int(v)
	}

	switch toSerialize.O {
	case pb.RowMajorContiguous:
	case pb.RowMajorNonContiguous:
		t.o = MakeDataOrder(NonContiguous)
	case pb.ColMajorContiguous:
		t.o = MakeDataOrder(ColMajor)
	case pb.ColMajorNonContiguous:
		t.o = MakeDataOrder(ColMajor, NonContiguous)
	}
	t.Δ = Triangle(toSerialize.T)
	typ := string(toSerialize.Type)
	for _, dt := range allTypes.set {
		if dt.String() == typ {
			t.t = dt
			break
		}
	}

	if t.e == nil {
		t.e = StdEng{}
	}
	t.makeArray(t.shape.TotalSize())

	// allocated data. Now time to actually copy over the data
	db := t.byteSlice()
	copy(db, toSerialize.Data)
	return t.sanity()
}
`

var (
	readNpy   *template.Template
	gobEncode *template.Template
	gobDecode *template.Template
	readCSV   *template.Template
)

func init() {
	readNpy = template.Must(template.New("readNpy").Funcs(funcs).Parse(readNpyRaw))
	readCSV = template.Must(template.New("readCSV").Funcs(funcs).Parse(readCSVRaw))
	gobEncode = template.Must(template.New("gobEncode").Funcs(funcs).Parse(gobEncodeRaw))
	gobDecode = template.Must(template.New("gobDecode").Funcs(funcs).Parse(gobDecodeRaw))
}

func generateDenseIO(f io.Writer, generic Kinds) {
	mk := Kinds{Kinds: filter(generic.Kinds, isNumber)}

	fmt.Fprintln(f, "/* GOB SERIALIZATION */\n")
	gobEncode.Execute(f, mk)
	gobDecode.Execute(f, mk)
	fmt.Fprint(f, "\n")

	fmt.Fprintln(f, "/* NPY SERIALIZATION */\n")
	fmt.Fprintln(f, npyDescRE)
	fmt.Fprintln(f, rowOrderRE)
	fmt.Fprintln(f, shapeRE)
	fmt.Fprintln(f, writeNpyRaw)
	readNpy.Execute(f, mk)
	fmt.Fprint(f, "\n")

	fmt.Fprintln(f, "/* CSV SERIALIZATION */\n")
	fmt.Fprintln(f, writeCSVRaw)
	readCSV.Execute(f, mk)
	fmt.Fprint(f, "\n")

	fmt.Fprintln(f, "/* FB SERIALIZATION */\n")
	fmt.Fprintln(f, fbEncodeDecodeRaw)
	fmt.Fprint(f, "\n")

	fmt.Fprintln(f, "/* PB SERIALIZATION */\n")
	fmt.Fprintln(f, pbEncodeDecodeRaw)
	fmt.Fprint(f, "\n")

}
