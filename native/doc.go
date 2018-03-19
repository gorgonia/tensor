// package native is a utility package for gorgonia.org/tensor.
//
// Amongst other things, it provides iterators that use Go slice semantics, while keeping a reference to the underlying memory.
// This means you can update the slices and the changes will be reflected back into the original tensor.
//
// There is of course a cost of using the native iterators and selectors - allocation costs.
// For best performance, don't use these in a tight loop.
package native
