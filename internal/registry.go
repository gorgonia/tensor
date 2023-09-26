package internal

import (
	"sync"

	"gorgonia.org/tensor/internal/errors"
)

// registry stuff is an incredibly bad pattern. However it's the only way that one can get things to work

// DataCaster is any function that converts a slice of byte into a slice of something else (arbitrarily listed as  `any`)
type DataCaster func([]byte) any

var engineRegistryLock sync.Mutex
var engineRegistry = map[string]Engine{
	"default": nil,
	"":        nil,
}

// RegisterEngine registers an engine
func RegisterEngine(e Engine, name string) error {
	engineRegistryLock.Lock()
	defer engineRegistryLock.Unlock()
	_, ok := engineRegistry[name]
	if ok {
		return errors.Errorf("Unable to register engine %q. Engine already exists", name)
	}
	engineRegistry[name] = e
	return nil
}

func LookupEngine(name string) Engine {
	engineRegistryLock.Lock()
	defer engineRegistryLock.Unlock()

	return engineRegistry[name]
}

var typeRegistry = map[string]DataCaster{
	"float64": bytes2Float64s,
}

func bytes2Float64s(buf []byte) any { panic("NYI") }
