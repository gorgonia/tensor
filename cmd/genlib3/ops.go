package main

type BinOp struct {
	Name          string
	Symbol        string
	TypeClass     string
	IsCommutative bool
	Interface     string // name of the interface that a method fulfils
}

var arithOps = []BinOp{
	{"Add", "+", "Addable", true, "Adder"},
	{"Sub", "-", "Num", false, "BasicArither"},
	{"Mul", "*", "Num", true, "BasicArither"},
	{"Div", "/", "Num", false, "BasicArither"},
}

var comparableOps = []BinOp{
	{"ElEq", "==", "comparable", false, "Comparer"},
	{"Ne", "!=", "comparable", false, "Comparer"},
}

var orderedOps = []BinOp{
	{"Lt", "<", "constraints.Ordered", false, "Ord"},
	{"Lte", "<=", "constraints.Ordered", false, "Ord"},
	{"Gt", ">", "constraints.Ordered", false, "FullOrd"},
	{"Gte", ">=", "constraints.Ordered", false, "FullOrd"},
}

var cmpOps = append(orderedOps, comparableOps...)

var cmpOpsNum = []BinOp{
	{"Lt", "<", "OrderedNum", false, "Ord"},
	{"Lte", "<=", "OrderedNum", false, "Ord"},
	{"Gt", ">", "OrderedNum", false, "FullOrd"},
	{"Gte", ">=", "OrderedNum", false, "FullOrd"},
	{"ElEq", "==", "Num", false, "Comparer"},
	{"Ne", "!=", "Num", false, "Comparer"},
}
