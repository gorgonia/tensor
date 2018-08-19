# Serialization #

This pseudopackage of sorts handles serialization. The "Canonical" serialized data structure is found in the `pb` subdirectory.

# Protobuf generation 

Proteus needs to be installed, as does its dependencies.


1. `cd pb`
2. `rm generated*`
3. `proteus -f ../../IDLs -p gorgonia.org/tensor/internal/serialization/pb`
4. `cd ../../IDLs`
5. `find gorgonia.org/ -mindepth 2 -type f -exec mv -i '{}' . ';'`
6. `rm -rf gorgonia.org`


# FlatBuffer generation
1. generate protobuf first
2. delete the `import "github.com/gogo/protobuf/gogoproto/gogo.proto";` line from the generated protobuf file
3. `flatc --proto PATH/TO/generated.proto`
4. place the `generated.fbs` file in the IDLs directory
4. restore the import line in the `generated.proto` file
5. From this directory: `flatc --go-namespace fb -g PATH/TO/generated.fbs`


# Notes #

`find gorgonia.org/ -mindepth 2 -type f -exec mv -i '{}' . ';'` is used to flatten  and put all the stuff in the root IDLs directory.

# The Serialization Story #

To serialize, we copy/convert/coerce the data to the internal/serialization data structures, then call the `Marshall` methods from there