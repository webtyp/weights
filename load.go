package weights

import (
	"webtyp.com/context"
	"webtyp.com/fetch"
	"webtyp.com/fmt"
	"webtyp.com/storage"
)

// QuotaEstimator is an optional interface that storage backends (e.g. IndexedDB)
// may implement to provide storage quota estimation.
type QuotaEstimator interface {
	EstimateQuota() (quota int64, usage int64, err error)
}

// LoadConfig configures artifact loading and storage caching.
type LoadConfig struct {
	ID         string
	Version    uint32
	URL        string
	Conn       storage.Conn
	OnProgress func(done, total int64)
	Fetcher    func(url string, cb func(*fetch.Response, error))
}

// CacheKey returns the storage key for an artifact ID and version.
func CacheKey(id string, version uint32) string {
	vStr := fmt.Convert(version).String()
	return fmt.Convert(id).WriteString("@").WriteString(vStr).String()
}

// Load fetches an artifact from URL or storage cache, verifies it, and opens it.
func Load(ctx *context.Context, cfg LoadConfig) (*Artifact, error) {
	key := CacheKey(cfg.ID, cfg.Version)

	// 1. Try reading from cache
	if cfg.Conn != nil {
		var cachedData []byte
		qRead := storage.Query{
			Action:     storage.ActionReadOne,
			Table:      "weights_cache",
			Columns:    []string{"data"},
			Conditions: []storage.Condition{storage.Eq("key", key)},
		}
		if _, err := cfg.Conn.Compile(qRead, nil); err == nil {
			scanner := cfg.Conn.QueryRow("cache_read")
			if err := scanner.Scan(&cachedData); err == nil && len(cachedData) > 0 {
				art, err := Open(cachedData)
				if err == nil {
					if cfg.OnProgress != nil {
						l := int64(len(cachedData))
						cfg.OnProgress(l, l)
					}
					return art, nil
				}
			}
		}
	}

	// 2. Fetch from network via webtyp.com/fetch
	var body []byte
	var fetchErr error
	doneChan := make(chan struct{})

	handleResp := func(resp *fetch.Response, err error) {
		if err != nil {
			fetchErr = err
		} else if resp == nil || resp.Status != 200 {
			fetchErr = ErrInvalidHeader
		} else {
			body = resp.Body()
			if cfg.OnProgress != nil {
				l := int64(len(body))
				cfg.OnProgress(l, l)
			}
		}
		close(doneChan)
	}

	if cfg.Fetcher != nil {
		cfg.Fetcher(cfg.URL, handleResp)
	} else {
		fetch.Get(cfg.URL).Send(handleResp)
	}

	<-doneChan
	if fetchErr != nil {
		return nil, fetchErr
	}

	// 3. Verify artifact BEFORE writing to cache
	art, err := Open(body)
	if err != nil {
		return nil, err
	}

	// 4. Cache ONLY if quota estimation succeeds and allows write
	if cfg.Conn != nil {
		if qe, ok := cfg.Conn.(QuotaEstimator); ok {
			quota, usage, qErr := qe.EstimateQuota()
			if qErr == nil && quota > 0 && usage+int64(len(body)) < quota {
				qWrite := storage.Query{
					Action:  storage.ActionCreate,
					Table:   "weights_cache",
					Columns: []string{"key", "data"},
					Values:  []any{key, body},
				}
				if _, err := cfg.Conn.Compile(qWrite, nil); err == nil {
					_ = cfg.Conn.Exec("cache_write")
				}
			}
		}
	}

	return art, nil
}

// Evict removes an artifact entry from storage cache by ID and version.
func Evict(conn storage.Conn, id string, version uint32) error {
	if conn == nil {
		return nil
	}
	key := CacheKey(id, version)
	qDel := storage.Query{
		Action:     storage.ActionDelete,
		Table:      "weights_cache",
		Conditions: []storage.Condition{storage.Eq("key", key)},
	}
	if _, err := conn.Compile(qDel, nil); err != nil {
		return err
	}
	return conn.Exec("cache_delete")
}
