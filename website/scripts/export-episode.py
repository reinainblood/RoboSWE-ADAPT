"""Publish one explicitly selected finalized episode, without provider payloads."""
import argparse
import hashlib
from html import escape as escape_html
import json
import re
import shutil
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboswe.review import export, atif


def sanitize(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key.lower() in {'authorization','api_key','api_key_env','access_token','headers','workspace_id','modal_profile','modal_sdk_python','profile','sdk_path','key'}:
                continue
            if isinstance(item,str) and (key in {'signature','encrypted_content'} or (key=='data' and ('encrypted' in str(value.get('type','')) or len(item)>100))):
                result[key] = {'opaque': True, 'notice': 'Opaque provider data omitted from public view; not readable reasoning.', 'sha256': hashlib.sha256(item.encode()).hexdigest()}
            else:
                result[key] = sanitize(item)
        return result
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if value.lstrip().startswith(('{','[')):
            try:return json.dumps(sanitize(json.loads(value)))
            except json.JSONDecodeError:pass
        value = value.replace(str(ROOT) + '/', '')
        value = re.sub(r'/Users/[^\s"\\<>]+', '[host path omitted]', value)
        value = re.sub(r'wrkspc_[A-Za-z0-9_-]+','[workspace omitted]',value)
        if re.search(r'\b(?:sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{25,})', value):
            raise ValueError('Potential credential in export')
        return value
    return value


def publication_metadata(job, review, label):
    if label not in {'development','incomplete-series','completed-series'}:
        raise ValueError('Explicit development, incomplete-series or completed-series publication label required')
    completion=None
    if label=='completed-series':
        from pairing.adapt_next_pairs import validate_series
        parent=ROOT/'jobs'/job['campaign_id']
        validate_series(parent,ROOT/'jobs')
        children=[json.loads(p.read_text()) for p in (ROOT/'jobs').glob('*/job.json')]
        children=sorted((c for c in children if c.get('campaign_id')==job['campaign_id'] and c.get('kind')=='adaptive-direct-attempt'),key=lambda c:c['attempt_number'])
        if job.get('id') not in {c['id'] for c in children}:raise ValueError('Selected episode does not belong to the completed series')
        completion={'series_id':job['campaign_id'],'verified_trials':5,'verified_successes':sum(c['status']=='verified_pass' for c in children)}
    if review.get('primary_leaderboard_eligible') is False:
        return {'classification':review.get('classification','development'), 'primary_leaderboard_eligible':False,
                'reason':review.get('reason','Excluded development series'), 'label':'Development episode — excluded from the primary leaderboard.'}
    if label=='development':
        return {'classification':'development','primary_leaderboard_eligible':False,'reason':'Explicitly published as development evidence.','label':'Development episode — excluded from the primary leaderboard.'}
    if completion is not None:
        return {'classification':'verified_individual_trial_in_completed_series','primary_leaderboard_eligible':None,
                'reason':'All five original trials in this series are independently replay verified. This page preserves the individual trial outcome; no comparative rank is claimed.',
                'label':'Verified individual trial · completed five-trial series.','series_completion':completion}
    return {'classification':'verified_individual_trial_in_incomplete_series','primary_leaderboard_eligible':None,
            'reason':'This individual trial is replay verified. The five-trial series is incomplete; no complete-series result or rank is claimed.',
            'label':'Verified individual trial · incomplete five-trial series.'}


def publish(job_id, destination, *, publication_label):
    source = ROOT / 'jobs' / job_id
    job = json.loads((source / 'job.json').read_text())
    if not job.get('finished_at') or job.get('status') not in {'verified_fail', 'verified_pass'}:
        raise ValueError('Only finalized independently verified attempts may be published')
    verification=json.loads((source/'verification.json').read_text())
    if verification.get('reproduced') is not True or verification.get('verified_success') is not (job['status']=='verified_pass'):
        raise ValueError('Finalized status must match independent replay receipt')
    review_path=ROOT/'jobs'/job['campaign_id']/'protocol-review.json'
    review=json.loads(review_path.read_text()) if review_path.exists() else {}
    publication=publication_metadata(job,review,publication_label)
    export(ROOT / 'jobs', destination, [job_id])
    target = destination / 'jobs' / job_id
    data = sanitize(json.loads((target / 'episode.json').read_text()))
    review_path=ROOT/'jobs'/job['campaign_id']/'protocol-review.json'
    review=json.loads(review_path.read_text()) if review_path.exists() else {}
    data['publication']=publication
    data['publication']['original_source_hashes_notice']='source_hashes identify original local receipts; exported JSON is a sanitized projection.'
    # The recorder emits fragmented MP4; browser metadata can expose only the
    # first fragment. Remux publication copies losslessly into a fast-start MP4.
    ffmpeg = next((ROOT / '.venv/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))
    video_provenance = []
    for view in data['videos']:
        media = target / view['name']
        original_hash = hashlib.sha256(media.read_bytes()).hexdigest()
        remuxed = media.with_suffix('.remux.mp4')
        subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-i', str(media), '-map', '0:v:0', '-c', 'copy', '-movflags', '+faststart', str(remuxed)], check=True)
        hashes=[]
        for candidate in (media,remuxed):
            hashes.append(subprocess.check_output([str(ffmpeg),'-v','error','-i',str(candidate),'-map','0:v:0','-f','hash','-hash','sha256','-'],text=True).strip())
        if hashes[0]!=hashes[1]:raise ValueError('Remux changed decoded video frames')
        remuxed.replace(media)
        video_provenance.append({'path':view['name'], 'original_sha256':original_hash, 'published_sha256':hashlib.sha256(media.read_bytes()).hexdigest(), 'decoded_frames_sha256':hashes[0], 'decoded_frames_match':True, 'transformation':'lossless container remux; video stream copied; faststart metadata'})
    data['publication']['video_provenance'] = video_provenance
    data['verification'] = sanitize(json.loads((source / 'verification.json').read_text()))
    # Only actually shipped assets appear in the public artifact inventory.
    data['job']['artifacts'] = {'episode': 'episode.json', 'atif': 'atif.json', **{v['name']:v['name'] for v in data['videos']}}
    data['reasoning_notice'] += ' Opaque provider blocks are labeled and represented by hashes, not rendered as readable reasoning.'
    (target / 'episode.json').write_text(json.dumps(data))
    (target / 'atif.json').write_text(json.dumps(atif(data)))
    records = sanitize(json.loads((destination / 'review-jobs.json').read_text()))
    (destination/'review-jobs.json').write_text(json.dumps(records))
    # HTTPS exports fetch the one canonical episode JSON instead of embedding
    # a second multi-megabyte copy. No file:// offline snapshot is promised.
    (destination/'snapshot.js').unlink()
    index = destination / 'index.html'
    html = index.read_text().replace('<script src="./snapshot.js"></script>', '').replace('RoboSWE-365 / Evidence', 'RoboSWE-ADAPT / Evidence').replace('RoboSWE-365 · Evidence', 'RoboSWE-ADAPT · Evidence')
    banner = '<div class="notice"><strong>'+escape_html(data['publication']['label'])+'</strong><br>'+escape_html(data['publication']['reason'])+' This page preserves the actual individual outcome, trajectory, provider-exposed reasoning, and four simulation-time camera views. <a href="../../#results">Return to results</a>.</div>'
    html = html.replace('<main>', '<main>' + banner)
    html = html.replace('ATIF projection</a>', 'ATIF-style projection</a>')
    html, notice_count = re.subn(r'(<button id="audit"[^>]*>.*?</button></div>)<p class="muted">.*?</p>', r'\1<p class="muted">Static selected-episode export. Paid judging is disabled. Recorded visual audits are shown below with their exact provider/model. Simulation-time videos omit thinking pauses; trajectory turns expose decision and control-step boundaries.</p>', html, count=1, flags=re.S)
    if notice_count != 1:
        raise ValueError('Review template changed: cannot locate static judge notice')
    html=html.replace('</style>', '.recorded-video{position:relative}.recorded-overlay{position:absolute;left:10px;right:10px;bottom:70px;background:#fffffff0;border:1px solid #d4d4d8;padding:10px;font:11px/1.4 ui-monospace,monospace;white-space:pre-wrap;pointer-events:none;max-height:125px;overflow:hidden}.overlay-toggle{display:block;margin:16px 0;font-size:12px}.overlay-toggle input{width:auto;margin:0 8px 0 0}@media(max-width:600px){.recorded-overlay{font-size:10px;max-height:100px}}</style>')
    html=html.replace('</body>','<script src="./episode-overlay.js" defer></script></body>')
    shutil.copy2(ROOT/'website/scripts/episode-overlay.js',destination/'episode-overlay.js')
    index.write_text(html)
    js = destination / 'review.js'
    content = js.read_text().replace("['Replay verified',r.verified_success===true?'Yes':r.verified_success===false?'No':'Unrecorded']", "['Replay outcome',data.verification?.reproduced===true?(data.verification.verified_success===true?'Verified pass':'Verified fail'):'Unverified']")
    content=content.replace('staticMode=false', 'staticMode=true').replace('if(!records.length){try{', 'if(!records.length&&!staticMode){try{')
    js.write_text(content)
    manifest = {'job_id':job_id, 'classification':data['publication'], 'files':[]}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            manifest['files'].append({'path':str(path.relative_to(destination)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest['total_bytes'] = sum(f['bytes'] for f in manifest['files'])
    (destination / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'directory':str(destination),'bytes':manifest['total_bytes'],'turns':len(data['turns']),'views':len(data['videos']),'reasoning_turns':sum(bool(t['reasoning']) for t in data['turns'])}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--publication-label',choices=['development','incomplete-series','completed-series'],required=True);a=p.parse_args()
    publish(a.job,a.destination,publication_label=a.publication_label)
