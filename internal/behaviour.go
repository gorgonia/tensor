package internal

type BroadcastBehaviour byte

const (
	NoBroadcast        BroadcastBehaviour = 0
	BroadcastShapeLeft                    = 1 << iota
	BroadcastShapeRight
	BroadcastData
	// if there's a new one then we gotta do 1<<iota
)

func (b BroadcastBehaviour) BroadcastData() bool {
	return b&BroadcastData == BroadcastData
}

func (b BroadcastBehaviour) BroadcastShape() BroadcastBehaviour {
	return (b & BroadcastShapeLeft) | (b & BroadcastShapeRight)
}
