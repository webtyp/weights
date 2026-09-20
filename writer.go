package weights

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"math"
)

// TensorInput represents a tensor to be written into an artifact.
type TensorInput struct {
	Name   string
	DType  DType
	Shape  []int
	Data   []byte
	Scales []float32
}

const Magic = "WTYPW1\x00\x00"

// WriteArtifact serializes the artifact parameters and tensor inputs into WTYPW1 binary format.
func WriteArtifact(id string, version uint32, tok TokenizerConfig, inputs []TensorInput) ([]byte, error) {
	if version == 0 {
		return nil, ErrBadVersion
	}

	// 1. Build raw tensor payload and relative tensor offsets (aligned to 64 bytes)
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
		tensorPayloadBuf.Write(inp.Data)
		currRelOffset += int64(len(inp.Data))
	}

	tensorPayload := tensorPayloadBuf.Bytes()
	checksum := crc32.ChecksumIEEE(tensorPayload)

	// 2. Measure binary header size with zeroed placeholders
	dummyHeader := encodeHeaderPayload(id, tok, inputs, relOffsets, checksum, 0, 0)
	headerLen := uint32(len(dummyHeader))

	payloadStart := (int64(16) + int64(headerLen) + 63) &^ 63
	totalLen := payloadStart + int64(len(tensorPayload))

	// 3. Encode final header payload
	finalHeader := encodeHeaderPayload(id, tok, inputs, relOffsets, checksum, totalLen, payloadStart)

	buf := new(bytes.Buffer)
	buf.WriteString(Magic)

	var verBytes [4]byte
	binary.LittleEndian.PutUint32(verBytes[:], version)
	buf.Write(verBytes[:])

	var hLenBytes [4]byte
	binary.LittleEndian.PutUint32(hLenBytes[:], uint32(len(finalHeader)))
	buf.Write(hLenBytes[:])

	buf.Write(finalHeader)

	padLen := payloadStart - (int64(16) + int64(len(finalHeader)))
	if padLen > 0 {
		buf.Write(make([]byte, padLen))
	}

	buf.Write(tensorPayload)

	return buf.Bytes(), nil
}

func encodeHeaderPayload(
	id string,
	tok TokenizerConfig,
	inputs []TensorInput,
	relOffsets []int64,
	checksum uint32,
	totalLen int64,
	payloadStart int64,
) []byte {
	buf := new(bytes.Buffer)

	writeString(buf, id)

	if tok.Lowercase {
		buf.WriteByte(1)
	} else {
		buf.WriteByte(0)
	}
	if tok.StripAccents {
		buf.WriteByte(1)
	} else {
		buf.WriteByte(0)
	}

	var u32 [4]byte
	var u64 [8]byte

	binary.LittleEndian.PutUint32(u32[:], uint32(len(tok.Vocab)))
	buf.Write(u32[:])
	for _, word := range tok.Vocab {
		writeString(buf, word)
	}

	binary.LittleEndian.PutUint32(u32[:], checksum)
	buf.Write(u32[:])

	binary.LittleEndian.PutUint64(u64[:], uint64(totalLen))
	buf.Write(u64[:])

	binary.LittleEndian.PutUint64(u64[:], uint64(payloadStart))
	buf.Write(u64[:])

	binary.LittleEndian.PutUint32(u32[:], uint32(len(inputs)))
	buf.Write(u32[:])

	for i, inp := range inputs {
		writeString(buf, inp.Name)

		buf.WriteByte(dtypeToByte(inp.DType))

		buf.WriteByte(byte(len(inp.Shape)))
		for _, dim := range inp.Shape {
			binary.LittleEndian.PutUint32(u32[:], uint32(dim))
			buf.Write(u32[:])
		}

		dataOffset := payloadStart + relOffsets[i]
		binary.LittleEndian.PutUint64(u64[:], uint64(dataOffset))
		buf.Write(u64[:])

		binary.LittleEndian.PutUint64(u64[:], uint64(len(inp.Data)))
		buf.Write(u64[:])

		binary.LittleEndian.PutUint32(u32[:], uint32(len(inp.Scales)))
		buf.Write(u32[:])

		for _, scale := range inp.Scales {
			bits := math.Float32bits(scale)
			binary.LittleEndian.PutUint32(u32[:], bits)
			buf.Write(u32[:])
		}
	}

	return buf.Bytes()
}

func writeString(buf *bytes.Buffer, s string) {
	var u16 [2]byte
	binary.LittleEndian.PutUint16(u16[:], uint16(len(s)))
	buf.Write(u16[:])
	buf.WriteString(s)
}
