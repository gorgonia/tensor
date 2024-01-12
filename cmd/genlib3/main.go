package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"os/user"
	"path"
	"runtime"
	"strings"
	//stdeng "gorgonia.org/tensor/engines"
)

const genmsg = "Code generated by genlib3. DO NOT EDIT"

var (
	gopath, tensorLoc, execLoc, denseLoc, stdengLoc string
)

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		usr, err := user.Current()
		if err != nil {
			log.Fatal(err)
		}
		gopath = path.Join(usr.HomeDir, "go")
		stat, err := os.Stat(gopath)
		if err != nil {
			log.Fatal(err)
		}
		if !stat.IsDir() {
			log.Fatal("You need to define a $GOPATH")
		}
	}
	tensorLoc = path.Join("src/gorgonia.org/tensor/internal/execution")
	denseLoc = "../../dense"
	execLoc = "../../internal/execution"
	stdengLoc = "../../engines"
}
func genExecutionArith(w io.Writer) {
	for _, op := range arithOps {
		executionArith.Execute(w, op)
	}
}

func genExecutionCmp(w io.Writer) {
	for _, op := range cmpOpsNum {
		executionCmp.Execute(w, op)
	}
	for _, op := range cmpOps {
		executionCmpBool.Execute(w, op)
	}
}

func genEnginesOps(w io.Writer) {
	for _, op := range cmpOps {
		enginesCmpBinOp.Execute(w, op)
	}
	for _, op := range cmpOpsNum {
		enginesOrderedNumOp.Execute(w, op)

	}
}
func genComparableEngMethods(w io.Writer) {
	for _, op := range comparableOps {
		compComparableEngMethods.Execute(w, op)
	}
}

func genOrderedEngMethods(w io.Writer) {
	for _, op := range orderedOps {
		orderedEngMethods.Execute(w, op)
	}
}

func genOrderedNumEngMethods(w io.Writer) {
	for _, op := range cmpOpsNum {
		orderedNumEngMethods.Execute(w, op)
	}
}

func genDenseArithPrepMethods(w io.Writer) {
	fmt.Fprintf(w, basicArithPrep)
}

func genDenseArithMethods(w io.Writer) {
	for _, op := range arithOps {
		denseArithOp.Execute(w, op)
	}
}

func genDenseCmpPrepMethods(w io.Writer) {
	fmt.Fprintf(w, basicCmpPrep)
}

func genDenseCmpMethods(w io.Writer) {
	for _, op := range cmpOpsNum {
		denseCmpOp.Execute(w, op)
	}
}
func genDenseArithMethodTests(w io.Writer) {

	for _, op := range arithOps {
		var writeTest bool
		if op.Identity != "" {
			idenTests.Execute(w, op)
			writeTest = true
		}
		if op.Inverse != "" {
			invTests.Execute(w, op)
			writeTest = true
		}
		if writeTest {
			datatypes := OrderedNum
			if op.Name == "Div" {
				datatypes = Floats
			}
			x := opDT{op, datatypes}
			denseArithMethodTest.Execute(w, x)
		}

	}
}
func genDenseCmpMethodTests(w io.Writer) {
	for _, op := range cmpOpsNum {
		transTests.Execute(w, op)
		x := opDT{op, OrderedNum}
		denseCmpMethodTest.Execute(w, x)
	}
}

func genStdEng()       {}
func genDenseMethods() {}

func writePkgName(f io.Writer, pkg string) {
	switch pkg {
	case stdengLoc:
		fmt.Fprintf(f, "// %s\n\npackage stdeng\n\n", genmsg)
	case tensorLoc:
		fmt.Fprintf(f, "// %s\n\npackage tensor\n\n", genmsg)
	case execLoc:
		fmt.Fprintf(f, "// %s\n\npackage execution\n\n", genmsg)
	case denseLoc:
		fmt.Fprintf(f, "// %s\n\n package dense\n\n", genmsg)
	default:
		fmt.Fprintf(f, "// %s\n\npackage unknown\n\n", genmsg)
	}
}

func pipeline(pkg, filename string, fns ...func(io.Writer)) {
	fullpath := path.Join(pkg, filename)
	f, err := os.Create(fullpath)
	if err != nil {
		log.Printf("fullpath %q", fullpath)
		log.Fatal(err)
	}
	defer f.Close()
	writePkgName(f, pkg)
	fmt.Fprint(f, "import \"gorgonia.org/tensor/internal/errors\"\n")

	for _, fn := range fns {
		fn(f)
	}

	// gofmt and goimports this stuff
	cmd := exec.Command("goimports", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, fullpath)
	}

	// account for differences in the postix from the linux sed
	if runtime.GOOS == "darwin" || strings.HasSuffix(runtime.GOOS, "bsd") {
		cmd = exec.Command("sed", "-i", "", `s/github.com\/alecthomas\/assert/github.com\/stretchr\/testify\/assert/g`, fullpath)
	} else {
		cmd = exec.Command("sed", "-E", "-i", `s/github.com\/alecthomas\/assert/github.com\/stretchr\/testify\/assert/g`, fullpath)
	}
	if err = cmd.Run(); err != nil {
		if err.Error() != "exit status 4" { // exit status 4 == not found
			log.Fatalf("sed failed with %v for %q", err.Error(), fullpath)
		}
	}

	cmd = exec.Command("gofmt", "-s", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Gofmt failed for %q", fullpath)
	}
}

func main() {
	pipeline(execLoc, "arith_gen.go", genExecutionArith)
	pipeline(execLoc, "cmp_gen.go", genExecutionCmp)
	pipeline(stdengLoc, "op_gen.go", genEnginesOps)
	pipeline(stdengLoc, "defaultComparableEngine_gen.go", genComparableEngMethods)
	pipeline(stdengLoc, "defaultOrderedEngine_gen.go", genOrderedEngMethods)
	pipeline(stdengLoc, "defaultOrderedNumEngine_gen.go", genOrderedNumEngMethods)

	pipeline(denseLoc, "arith.go", genDenseArithPrepMethods, genDenseArithMethods)
	pipeline(denseLoc, "cmp.go", genDenseCmpPrepMethods, genDenseCmpMethods)
	pipeline(denseLoc, "arith_gen_test.go", genDenseArithMethodTests)
	pipeline(denseLoc, "cmp_gen_test.go", genDenseCmpMethodTests)
	genStdEng()
	genDenseMethods()
}
