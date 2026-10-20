import json
import sys
from unittest.mock import patch

import pytest

from roboswe.review import episode, main


def test_old_failed_task_agreement_projection(tmp_path):
    (tmp_path/'job.json').write_text(json.dumps({'id': 'test', 'status': 'finished'}))
    (tmp_path/'report.json').write_text('{}')
    audit_dir = tmp_path/'predicate-audits'/'old'; audit_dir.mkdir(parents=True)
    path = audit_dir/'audit.json'
    original = json.dumps({'status': 'ok', 'review_status': 'contradicted', 'mechanical_success': False, 'answer': {'visible_success': False}})
    path.write_text(original)
    audit = episode(tmp_path)['audits'][0]
    assert audit['comparison_status'] == 'consistent_failure'
    assert not audit['predicate_disagreement']
    assert path.read_text() == original


def test_enable_requires_explicit_ledger_and_cap():
    with patch.object(sys, 'argv', ['review', '--enable-judge']), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
