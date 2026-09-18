package weights

import (
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
)

const Magic = "WTYPW1\x00\x00"

// HeaderTensor holds metadata and location of a tensor inside the artifact file.
type HeaderTensor struct {
	Name       string    `json:"name"`
	DType      DType     `json:"dtype"`
	Shape      []int     `json:"shape"`
	DataOffset int64     `json:"data_offset"`
	DataLen    int64     `json:"data_len"`
	Scales     []float32 `json:"scales,omitempty"`
}

// Header represents the JSON metadata stored at the beginning of the artifact.
type Header struct {
	ID        string          `json:"id"`
	Version   uint32          `json:"version"`
	Tokenizer TokenizerConfig `json:"tokenizer"`
	Tensors   []HeaderTensor  `json:"tensors"`
	TotalLen  int64           `json:"total_len,omitempty"`
	Checksum  uint32          `json:"checksum,omitempty"`
}

// Open reads an artifact from a byte slice. It does not copy tensor data:
// the returned Artifact aliases src, which must outlive it.
func Open(src []byte) (*Artifact, error) {
	if len(src) < 16 {
		return nil, ErrTruncatedTensorData
	}

	if string(src[0:8]) != Magic {
		return nil, ErrBadMagic
	}

	version := binary.LittleEndian.Uint32(src[8:12])
	if version == 0 {
		return nil, ErrBadVersion
	}

	headerLen := binary.LittleEndian.Uint32(src[12:16])
	if int64(16)+int64(headerLen) > int64(len(src)) {
		return nil, ErrTruncatedTensorData
	}

	headerBytes := src[16 : 16+headerLen]
	var hdr Header
	if err := json.Unmarshal(headerBytes, &hdr); err != nil {
		return nil, ErrInvalidHeader
	}

	if hdr.TotalLen > 0 && int64(len(src)) < hdr.TotalLen {
		return nil, ErrTruncatedTensorData
	}

	payloadStart := int64(16) + int64(headerLen)
	if len(hdr.Tensors) > 0 && hdr.Tensors[0].DataOffset >= payloadStart {
		payloadStart = hdr.Tensors[0].DataOffset
	}

	payloadEnd := int64(len(src))
	if hdr.TotalLen > 0 && hdr.TotalLen < payloadEnd {
		payloadEnd = hdr.TotalLen
	}

	if hdr.Checksum != 0 {
		calculated := crc32.ChecksumIEEE(src[payloadStart:payloadEnd])
		if calculated != hdr.Checksum {
			return nil, ErrChecksumMismatch
		}
	}

	tensors := make([]Tensor, len(hdr.Tensors))
	for i, ht := range hdr.Tensors {
		if ht.DataOffset%64 != 0 {
			return nil, ErrBadAlignment
		}
		if ht.DataOffset < payloadStart || ht.DataOffset+ht.DataLen > int64(len(src)) {
			return nil, ErrTruncatedTensorData
		}

		tensors[i] = Tensor{
			Name:   ht.Name,
			DType:  ht.DType,
			Shape:  ht.Shape,
			Data:   src[ht.DataOffset : ht.DataOffset+ht.DataLen],
			Scales: ht.Scales,
		}
	}

	return &Artifact{
		ID:        hdr.ID,
		Version:   hdr.Version,
		Tensors:   tensors,
		Tokenizer: hdr.Tokenizer,
	}, nil
}
