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
