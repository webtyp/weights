package weights

// QuantizeInt4Block32 quantizes one row (len(row) a multiple of BlockSize) the way GGUF's Q4_0
// does: per block, max is the value of largest magnitude with its sign, scale = max / −8,
// nibble = min(15, int(x/scale + 8.5)) (all 8 when scale is 0). It returns len(row)/2 bytes and
// len(row)/BlockSize scales.
func QuantizeInt4Block32(row []float32) (q []byte, scales []float32, err error) {
	if len(row)%BlockSize != 0 {
		return nil, nil, ErrInt4Cols
	}
	blocks := len(row) / BlockSize
	q = make([]byte, blocks*Int4BlockBytes)
	scales = make([]float32, blocks)
	for b := 0; b < blocks; b++ {
		x := row[b*BlockSize : (b+1)*BlockSize]
		var max, amax float32
		for _, v := range x {
			a := v
			if a < 0 {
				a = -a
			}
			if a > amax {
				amax, max = a, v
			}
		}
		scale := max / -8
		var inv float32
		if scale != 0 {
			inv = 1 / scale
		}
		scales[b] = scale
		out := q[b*Int4BlockBytes : (b+1)*Int4BlockBytes]
		for j := 0; j < Int4BlockBytes; j++ {
			out[j] = nibble(x[j]*inv) | nibble(x[j+Int4BlockBytes]*inv)<<4
		}
	}
	return q, scales, nil
}

// nibble maps v = x/scale, in [−8, 8], to its 4-bit code: min(15, int(v + 8.5)).
func nibble(v float32) byte {
	n := int8(v + 8.5)
	if n > 15 {
		n = 15
	}
	return byte(n)
}

// DequantInt4Block32 writes the len(q)*2 values of q (one row, or any whole number of blocks)
// into dst: dst[b*32+j] = (q[b*16+j]&0x0F − 8) × scales[b], dst[b*32+j+16] = (q[b*16+j]>>4 − 8) × scales[b].
func DequantInt4Block32(dst []float32, q []byte, scales []float32) {
	blocks := len(q) / Int4BlockBytes
	for b := 0; b < blocks; b++ {
		s := scales[b]
		in := q[b*Int4BlockBytes : (b+1)*Int4BlockBytes]
		out := dst[b*BlockSize : (b+1)*BlockSize]
		for j, v := range in {
			out[j] = float32(int(v&0x0F)-8) * s
			out[j+Int4BlockBytes] = float32(int(v>>4)-8) * s
		}
	}
}
