package weights

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
)

// TensorInput represents a tensor to be written into an artifact.
type TensorInput struct {
	Name   string
	DType  DType
	Shape  []int
	Data   []byte
	Scales []float32
}

// WriteArtifact serializes the given artifact parameters and tensor inputs into WTYPW1 format bytes.
func WriteArtifact(id string, version uint32, tok TokenizerConfig, inputs []TensorInput) ([]byte, error) {
	if version == 0 {
		return nil, ErrBadVersion
	}

	headerTensors := make([]HeaderTensor, len(inputs))
	for i, inp := range inputs {
		headerTensors[i] = HeaderTensor{
			Name:   inp.Name,
			DType:  inp.DType,
			Shape:  inp.Shape,
			Scales: inp.Scales,
		}
	}

	hdr := Header{
		ID:        id,
		Version:   version,
		Tokenizer: tok,
		Tensors:   headerTensors,
		TotalLen:  0,
		Checksum:  0,
	}

	// 1. Build tensor payload bytes and determine tensor data offsets relative to payload start
	tensorPayloadBuf := new(bytes.Buffer)
	relOffsets := make([]int64, len(inputs))

	var currRelOffset int64
	for i, inp := range inputs {
		alignedRel := (currRelOffset + 63) &^ 63
		if pad := alignedRel - currRelOffset; pad > 0 {
			tensorPayloadBuf.Write(make([]byte, pad))
			currRelOffset += pad
		}
		relOffsets[i] = currRelOffset
		hdr.Tensors[i].DataLen = int64(len(inp.Data))
		tensorPayloadBuf.Write(inp.Data)
		currRelOffset += int64(len(inp.Data))
	}

	tensorPayload := tensorPayloadBuf.Bytes()
	hdr.Checksum = crc32.ChecksumIEEE(tensorPayload)

	// 2. Determine payloadStart and final JSON header bytes
	var finalJsonBytes []byte
	var payloadStart int64

	for {
		for i := range inputs {
			hdr.Tensors[i].DataOffset = payloadStart + relOffsets[i]
		}
		hdr.TotalLen = payloadStart + int64(len(tensorPayload))

		marshaled, err := json.Marshal(hdr)
		if err != nil {
			return nil, ErrInvalidHeader
		}

		newPayloadStart := (16 + int64(len(marshaled)) + 63) &^ 63
		if newPayloadStart == payloadStart && len(marshaled) == len(finalJsonBytes) {
			break
		}
		payloadStart = newPayloadStart
		finalJsonBytes = marshaled
	}

	// 3. Assemble final binary artifact
	buf := new(bytes.Buffer)
	buf.WriteString(Magic)

	var verBytes [4]byte
	binary.LittleEndian.PutUint32(verBytes[:], version)
	buf.Write(verBytes[:])

	var hLenBytes [4]byte
	binary.LittleEndian.PutUint32(hLenBytes[:], uint32(len(finalJsonBytes)))
	buf.Write(hLenBytes[:])

	buf.Write(finalJsonBytes)

	padLen := payloadStart - (16 + int64(len(finalJsonBytes)))
	if padLen > 0 {
		buf.Write(make([]byte, padLen))
	}

	buf.Write(tensorPayload)

	return buf.Bytes(), nil
}
