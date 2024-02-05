package main

type BinOp struct {
	Name          string
	Symbol        string
	TypeClass     string
	IsCommutative bool
	Interface     string // name of the interface that a method fulfils

	Identity string // idenitty element. It's a string here so that a comparison against "" can be made
	Inverse  string // inverse operation
}

var arithOps = []BinOp{
	{"Add", "+", "Addable", true, "Adder", "0", ""},
	{"Sub", "-", "Num", false, "BasicArither", "", "Add"},
	{"Mul", "*", "Num", true, "BasicArither", "1", ""},
	{"Div", "/", "Num", false, "BasicArither", "", "Mul"},
}

var comparableOps = []BinOp{
	{"ElEq", "==", "comparable", false, "Comparer", "", ""},
	{"Ne", "!=", "comparable", false, "Comparer", "", ""},
}

var orderedOps = []BinOp{
	{"Lt", "<", "constraints.Ordered", false, "Ord", "", ""},
	{"Lte", "<=", "constraints.Ordered", false, "Ord", "", ""},
	{"Gt", ">", "constraints.Ordered", false, "FullOrd", "", ""},
	{"Gte", ">=", "constraints.Ordered", false, "FullOrd", "", ""},
}

var cmpOps = append(orderedOps, comparableOps...)

var cmpOpsNum = []BinOp{
	{"Lt", "<", "OrderedNum", false, "Ord", "", ""},
	{"Lte", "<=", "OrderedNum", false, "Ord", "", ""},
	{"Gt", ">", "OrderedNum", false, "FullOrd", "", ""},
	{"Gte", ">=", "OrderedNum", false, "FullOrd", "", ""},
	{"ElEq", "==", "Num", false, "Comparer", "", ""},
	{"Ne", "!=", "Num", false, "Comparer", "", ""},
}

type opDT struct {
	BinOp
	Datatypes []string
}

type UnOp struct {
	Name      string
	TypeClass string
	Interface string
	PkgFunc   string
	Body      string
}

var unops = []UnOp{
	{"Abs", "Num", "Abser", "Abs", ""},
	{"Sign", "Num", "Signer", "", `if x < 0 { return -1 }; return 1`},
	{"Ceil", "Floats", "Ceiler", "Ceil", ""},
	{"Floor", "Floats", "Floorer", "Floor", ""},
	{"Neg", "Num", "Neger", "", "return -x"},
	{"Inv", "Num", "Inver", "", "return 1/x"},
	{"InvSqrt", "Floats", "InvSqrter", "", "return 1/{{.MathPkg}}.Sqrt(x)"},
	{"Exp", "Floats", "ExpLoger", "Exp", ""},
	{"Log", "Floats", "ExpLoger", "Log", ""},
	{"Log2", "Floats", "ExpLoger", "Log2", ""},
	{"Log10", "Floats", "ExpLoger", "Log10", ""},
	{"Log1p", "Floats", "ExpLoger", "Log1p", ""},
	{"Expm1", "Floats", "ExpLoger", "Expm1", ""},
	{"Square", "Num", "Squarer", "", "return x*x"},
	{"Sqrt", "Floats", "Squarer", "Sqrt", ""},
	{"Cube", "Num", "Cuber", "", "return x*x*x"},
	{"Tanh", "Floats", "Tanher", "Tanh", ""},
}

type FloatEng struct {
	Type       string
	EngineName string
	MathPkg    string
}

var floatEngs = []FloatEng{
	{"float64", "StdFloat64Engine", "math"},
	{"float32", "StdFloat32Engine", "math32"},
}

type FloatEngUnOp struct {
	UnOp
	FloatEng
}

var floatengUnOps []FloatEngUnOp

func init() {
	for _, e := range floatEngs {
		for _, o := range unops {
			floatengUnOps = append(floatengUnOps, FloatEngUnOp{o, e})
		}
	}
}
