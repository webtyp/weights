package weights

import (
	"context"
	"io"
	"net/http"
	"strconv"
)

// StorageConn abstracts storage operations (IndexedDB in browser or key-value store).
type StorageConn interface {
	Get(key string) ([]byte, error)
	Put(key string, val []byte) error
	Delete(key string) error
	EstimateQuota() (quota int64, usage int64, err error)
}

// Fetcher abstracts HTTP fetching to facilitate testing or WASM wrappers.
type Fetcher interface {
	Fetch(ctx context.Context, url string, onProgress func(done, total int64)) ([]byte, error)
}

type defaultFetcher struct{}

func (f defaultFetcher) Fetch(ctx context.Context, url string, onProgress func(done, total int64)) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, Error("weights: http status " + strconv.Itoa(resp.StatusCode))
	}

	total := resp.ContentLength
	var done int64
	var buf []byte
	if total > 0 {
		buf = make([]byte, 0, total)
	}

	tmp := make([]byte, 32*1024)
	for {
		n, rErr := resp.Body.Read(tmp)
		if n > 0 {
			buf = append(buf, tmp[:n]...)
			done += int64(n)
			if onProgress != nil {
				onProgress(done, total)
			}
		}
		if rErr != nil {
			if rErr == io.EOF {
				break
			}
			return nil, rErr
		}
	}

	return buf, nil
}

// LoadConfig configures artifact loading and browser caching.
type LoadConfig struct {
	ID         string
	Version    uint32
	URL        string
	Conn       StorageConn
	Fetcher    Fetcher
	OnProgress func(done, total int64)
}

func cacheKey(id string, version uint32) string {
	return id + "@" + strconv.FormatUint(uint64(version), 10)
}

// Load fetches an artifact from URL or storage cache, validates it, and opens it.
func Load(ctx context.Context, cfg LoadConfig) (*Artifact, error) {
	key := cacheKey(cfg.ID, cfg.Version)

	// Check storage cache
	if cfg.Conn != nil {
		if cached, err := cfg.Conn.Get(key); err == nil && len(cached) > 0 {
			art, err := Open(cached)
			if err == nil {
				if cfg.OnProgress != nil {
					cfg.OnProgress(int64(len(cached)), int64(len(cached)))
				}
				return art, nil
			}
		}
	}

	fetcher := cfg.Fetcher
	if fetcher == nil {
		fetcher = defaultFetcher{}
	}

	data, err := fetcher.Fetch(ctx, cfg.URL, cfg.OnProgress)
	if err != nil {
		return nil, err
	}

	// Verify before caching or returning
	art, err := Open(data)
	if err != nil {
		return nil, err
	}

	// Store in cache after verification if quota allows
	if cfg.Conn != nil {
		quota, usage, err := cfg.Conn.EstimateQuota()
		if err == nil && quota > 0 {
			if usage+int64(len(data)) < quota {
				_ = cfg.Conn.Put(key, data)
			}
		} else {
			// If quota estimation is not supported or returns 0, try put anyway
			_ = cfg.Conn.Put(key, data)
		}
	}

	return art, nil
}

// Evict removes an artifact entry from storage cache by key or ID pattern.
func Evict(conn StorageConn, key string) error {
	if conn == nil {
		return nil
	}
	return conn.Delete(key)
}
