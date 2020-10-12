package paginated

import "fmt"

var (
	FormatCSV     = "csv"
	FormatJSON    = "json"
	FormatGob     = "gob"
	FormatProto   = "protbuf"
	FormatFlat    = "flatbuf"
	FormatMsgPck  = "msgpack"
	FormatParquet = "parquet"
	FormatNumpy   = "numpy"
)

// Format does not actually do anything for now.
func (p *Tensor) Format(f fmt.State, c rune) {
	return
}

// String returns an empty string.
func (p *Tensor) String() string {
	return ""
}

func (p *Tensor) fileExtension() string {
	// supported
	switch p.fileFormat {
	case FormatCSV:
		return "csv"
	case FormatJSON:
		return "json"
	case FormatGob:
		return "gob"
	case FormatProto:
		return "proto"
	case FormatFlat:
		return "flat"
	case FormatMsgPck:
		return "msgpack"
	case FormatParquet:
		return "parquet"
	case FormatNumpy:
		return ".npy"
	}

	// unsupported
	return ""
}
