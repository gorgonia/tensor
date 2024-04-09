package paginated

// Stream is an interface for streaming data
// to a paginated tensor.
type Stream struct {
	*Tensor
	data  chan Array
	close chan bool
}

// Stream can be used to more easily write a large amount of data
// to an initialized paginated tensor.
func (p *Tensor) Stream(mutable []bool) (*Stream, error) {
	if len(mutable) != p.Dims() {
		return &Stream{}, ErrSize
	}

	dataChan := make(chan Array, 1)
	closeChan := make(chan bool, 1)

	go func(p *Tensor) {
		var b bool
		for {
			select {
			case data := <-dataChan:
				p.addData(data)
			case <-closeChan:
				close(dataChan)
				close(closeChan)
				b = true
			}

			if b {
				break
			}
		}
	}(p)

	return &Stream{
		Tensor: p,
		data:   dataChan,
		close:  closeChan,
	}, nil
}

// Send will block until the Stream is ready
// to accept and write new data to the paginated tensor.
func (s *Stream) Send(data Array) {
	s.data <- data
}

// Close will close all channels, thus killing the Stream
func (s *Stream) Close() {
	s.close <- true
}

func (p *Tensor) addData(data Array) error {
	page, ok := p.firstUnfilledPage()
	if !ok {
		return ErrFilled
	}

	var pageData Array
	d, ok := p.cache.Get(page.Id)
	if !ok {
		err := p.Swap(page.Id)
		if err != nil {
			return err
		}

		d, ok = p.cache.Get(page.Id)
		if !ok {
			return ErrCache
		}
	}
	pageData = d.(Array)

	// remaining page-capacity
	if page.Datasize > 0 {
		var err error
		pageData, err = pageData.Slice(page.Datasize-1, pageData.Len()-1)
		if err != nil {
			return err
		}
	}

	// slice of new data
	capacity := pageData.Len()
	if data.Len() < capacity {
		capacity = data.Len()
	}
	slice, err := data.Slice(0, capacity)
	if err != nil {
		return err
	}

	// copy data
	err = pageData.Copy(slice)
	if err != nil {
		return err
	}
	page.Datasize += slice.Len()

	// recurse: for all data in supplied array
	if slice.Len() < data.Len() {
		remaining, err := data.Slice(slice.Len(), data.Len()-1)
		if err != nil {
			return err
		}

		return p.addData(remaining)
	}

	return nil
}
