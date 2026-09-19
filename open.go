package weights

import (
	"encoding/binary"
	"hash/crc32"
	"math"
)

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
	headerEnd := int64(16) + int64(headerLen)
	if headerEnd > int64(len(src)) {
		return nil, ErrTruncatedTensorData
	}

	pos := 16
	readString := func() (string, error) {
		if pos+2 > int(headerEnd) {
			return "", ErrTruncatedTensorData
		}
		sLen := int(binary.LittleEndian.Uint16(src[pos : pos+2]))
		pos += 2
		if pos+sLen > int(headerEnd) {
			return "", ErrTruncatedTensorData
		}
		s := string(src[pos : pos+sLen])
		pos += sLen
		return s, nil
	}

	readByte := func() (byte, error) {
		if pos+1 > int(headerEnd) {
			return 0, ErrTruncatedTensorData
		}
		b := src[pos]
		pos++
		return b, nil
	}

	readU32 := func() (uint32, error) {
		if pos+4 > int(headerEnd) {
			return 0, ErrTruncatedTensorData
		}
		val := binary.LittleEndian.Uint32(src[pos : pos+4])
		pos += 4
		return val, nil
	}

	readI64 := func() (int64, error) {
		if pos+8 > int(headerEnd) {
			return 0, ErrTruncatedTensorData
		}
		val := int64(binary.LittleEndian.Uint64(src[pos : pos+8]))
		pos += 8
		return val, nil
	}

	id, err := readString()
	if err != nil {
		return nil, err
	}

	lowercaseByte, err := readByte()
	if err != nil {
		return nil, err
	}
	stripAccentsByte, err := readByte()
	if err != nil {
		return nil, err
	}

	vocabCount, err := readU32()
	if err != nil {
		return nil, err
	}

	vocab := make([]string, vocabCount)
	for i := uint32(0); i < vocabCount; i++ {
		w, err := readString()
		if err != nil {
			return nil, err
		}
		vocab[i] = w
	}

	checksum, err := readU32()
	if err != nil {
		return nil, err
	}

	totalLen, err := readI64()
	if err != nil {
		return nil, err
	}

	payloadStart, err := readI64()
	if err != nil {
		return nil, err
	}

	// Mandatory verification checks (checksum and totalLen cannot be 0 or omitted)
	if checksum == 0 || totalLen <= 0 {
		return nil, ErrInvalidHeader
	}

	if int64(len(src)) < totalLen {
		return nil, ErrTruncatedTensorData
	}

	if payloadStart < headerEnd || payloadStart > totalLen {
		return nil, ErrInvalidHeader
	}

	calculated := crc32.ChecksumIEEE(src[payloadStart:totalLen])
	if calculated != checksum {
		return nil, ErrChecksumMismatch
	}

	tensorCount, err := readU32()
	if err != nil {
		return nil, err
	}

	tensors := make([]Tensor, tensorCount)
	for i := uint32(0); i < tensorCount; i++ {
		tName, err := readString()
		if err != nil {
			return nil, err
		}

		dtByte, err := readByte()
		if err != nil {
			return nil, err
		}
		dt, err := byteToDType(dtByte)
		if err != nil {
			return nil, err
		}

		shapeLenByte, err := readByte()
		if err != nil {
			return nil, err
		}
		shape := make([]int, shapeLenByte)
		for s := byte(0); s < shapeLenByte; s++ {
			dim, err := readU32()
			if err != nil {
				return nil, err
			}
			shape[s] = int(dim)
		}

		dataOffset, err := readI64()
		if err != nil {
			return nil, err
		}

		dataLen, err := readI64()
		if err != nil {
			return nil, err
		}

		scalesCount, err := readU32()
		if err != nil {
			return nil, err
		}
		var scales []float32
		if scalesCount > 0 {
			scales = make([]float32, scalesCount)
			for sc := uint32(0); sc < scalesCount; sc++ {
				bits, err := readU32()
				if err != nil {
					return nil, err
				}
				scales[sc] = math.Float32frombits(bits)
			}
		}

		if dataOffset%64 != 0 {
			return nil, ErrBadAlignment
		}
		if dataOffset < payloadStart || dataOffset+dataLen > totalLen {
			return nil, ErrTruncatedTensorData
		}

		tensors[i] = Tensor{
			Name:   tName,
			DType:  dt,
			Shape:  shape,
			Data:   src[dataOffset : dataOffset+dataLen],
			Scales: scales,
		}
	}

	return &Artifact{
		ID:      id,
		Version: version,
		Tensors: tensors,
		Tokenizer: TokenizerConfig{
			Lowercase:    lowercaseByte != 0,
			StripAccents: stripAccentsByte != 0,
			Vocab:        vocab,
		},
	}, nil
}
