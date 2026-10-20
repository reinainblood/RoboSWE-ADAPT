"""Copy the replay-gated snapshot for publication, stripping host-specific paths."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_EXECUTION_FIELDS = {
    'backend', 'simulator_image', 'code_image', 'modal_environment', 'artifact_transfer',
}


def sanitize(value):
    if isinstance(value, dict):
        return {
            key: sanitize({k: v for k, v in item.items() if k in PUBLIC_EXECUTION_FIELDS}
                          if key == 'execution' and isinstance(item, dict) else item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        try:
            return str(Path(value).relative_to(ROOT))
        except ValueError:
            raise ValueError('Refusing to publish an absolute path outside the project') from None
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    data = sanitize(json.loads(args.snapshot.read_text()))
    if data.get('schema_version') != 1 or data.get('protocol') != 'adaptive-public-sqlite-v1':
        raise ValueError('Expected public adaptive results schema v1')
    output = ROOT / 'website/assets/adaptive-results.json'
    output.write_text(json.dumps(data, indent=2) + '\n')
    print(output)
