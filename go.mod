module gorgonia.org/tensor

go 1.20

replace gorgonia.org/dtype => ../dtype

replace gorgonia.org/shapes => ../shapes

replace gorgonia.org/vecf64 => ../vecf64

replace gorgonia.org/vecf32 => ../vecf32

require (
	github.com/chewxy/math32 v1.10.1
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.8.3
	golang.org/x/exp v0.0.0-20230522175609-2e198f4a06a1
	gonum.org/v1/gonum v0.13.0
	google.golang.org/protobuf v1.30.0
	gorgonia.org/dtype v0.10.0
	gorgonia.org/shapes v0.0.0-20220805023001-db33330e8e09
	gorgonia.org/vecf32 v0.9.0
	gorgonia.org/vecf64 v0.9.0
)

require (
	github.com/chewxy/hm v1.0.0 // indirect
	github.com/davecgh/go-spew v1.1.1 // indirect; indirec
	github.com/google/gofuzz v1.2.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/xtgo/set v1.0.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
