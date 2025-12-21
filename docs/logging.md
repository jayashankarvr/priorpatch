# Logging Configuration

PriorPatch uses Python's standard logging module. Here's how to configure it for your needs.

## CLI Logging

Control log verbosity with the `--log-level` flag:

```bash
# Minimal output (errors only)
priorpatch analyze --input image.jpg --outdir results/ --log-level ERROR

# Normal output (default)
priorpatch analyze --input image.jpg --outdir results/ --log-level INFO

# Verbose output (for debugging)
priorpatch analyze --input image.jpg --outdir results/ --log-level DEBUG
```

## Python API Logging

### Basic Setup

```python
import logging
from priorpatch import Ensemble, load_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Use PriorPatch
ensemble = Ensemble.from_config('config/detectors.json')
img = load_image('photo.jpg')
result = ensemble.score_image(img)
```

### Save Logs to File

```python
import logging

# Log to file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='priorpatch.log',
    filemode='w'
)

# Or log to both file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler('priorpatch.log')
fh.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
```

### Filter by Component

```python
import logging

# Only show logs from specific detectors
logging.getLogger('priorpatch.detectors.color_stats').setLevel(logging.DEBUG)
logging.getLogger('priorpatch.detectors.fft_dct').setLevel(logging.WARNING)

# Silence a noisy detector
logging.getLogger('priorpatch.detectors.copy_move').setLevel(logging.ERROR)

# Core ensemble logs only
logging.getLogger('priorpatch.core').setLevel(logging.INFO)
```

## Log Levels Guide

| Level | When to Use | What You'll See |
|-------|-------------|-----------------|
| **DEBUG** | Debugging issues | Detailed info: patch-by-patch processing, detector computations |
| **INFO** | Normal operation | Progress updates, completion messages |
| **WARNING** | Important notices | Detector failures, unusual inputs, performance issues |
| **ERROR** | Problems | Critical failures, invalid inputs, unrecoverable errors |

## What Gets Logged

### DEBUG Level

- Individual patch processing
- Detector score calculations
- Image conversions
- Normalization details

### INFO Level

- Image loading
- Ensemble initialization
- Number of patches to process
- Heatmap generation
- File I/O operations

### WARNING Level

- Image size mismatches
- Detector failures (<50% patches)
- Path traversal attempts
- Config inconsistencies

### ERROR Level

- Missing files
- Invalid configurations
- High detector failure rates (>50%)
- Critical processing errors

## Examples

### Debug a Specific Image

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from priorpatch import Ensemble, load_image

ensemble = Ensemble.from_config('config/detectors.json')
img = load_image('problematic_image.jpg')
result = ensemble.score_image(img)

# Check logs for any warnings or errors
```

### Quiet Mode (Only Errors)

```python
import logging

logging.basicConfig(level=logging.ERROR)

# Now PriorPatch will only show errors, no progress updates
```

### Custom Format

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',  # Simple format
    datefmt='%H:%M:%S'
)
```

## Troubleshooting

### No Logs Appearing?

If you don't see any logs:

1. Check that logging is configured **before** importing priorpatch
2. Make sure log level is not set too high (ERROR will hide INFO messages)
3. Verify no other library is overriding the logging config

### Too Much Output?

```python
# Reduce verbosity
logging.getLogger('priorpatch').setLevel(logging.WARNING)

# Or disable progress bars
import os
os.environ['TQDM_DISABLE'] = '1'
```

### Performance Impact

Logging has minimal performance impact:

- **INFO level**: < 1% overhead
- **DEBUG level**: 2-5% overhead (due to detailed per-patch logs)

For production use, stick with INFO or WARNING level.
