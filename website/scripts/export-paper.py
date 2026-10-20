"""Render the committed methods draft as accessible HTML, with linked references."""
import argparse,hashlib,html,json,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--pandoc',default='pandoc');p.add_argument('--source',type=Path,default=ROOT/'paper/main.tex');a=p.parse_args()
 source=a.source.resolve();text=source.read_text()
 bibliography=re.search(r'\\begin\{thebibliography\}\{[^}]+\}(.*?)\\end\{thebibliography\}',text,re.S)
 if not bibliography:raise ValueError('Expected explicit bibliography')
 entries=re.findall(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem\{|\Z)',bibliography.group(1),re.S)
 ids={key:str(index+1) for index,(key,_) in enumerate(entries)}
 def convert(value,standalone=False):
  command=[a.pandoc,'--from=latex','--to=html5','--mathml','--shift-heading-level-by=1']
  if standalone:command+=['--standalone','--metadata','lang=en']
  return subprocess.check_output(command,input=value,text=True)
 converted=convert(text[:bibliography.start()]+text[bibliography.end():],True)
 body=re.search(r'<body>(.*?)</body>',converted,re.S).group(1)
 def cite(match):
  keys=match.group(1).split()
  return '<span class="citation">'+', '.join(f'<a href="#ref-{html.escape(k)}" aria-label="Reference {ids[k]}">[{ids[k]}]</a>' for k in keys)+'</span>'
 body=re.sub(r'<span\s+class="citation"\s+data-cites="([^"]+)"\s*>.*?</span>',cite,body,flags=re.S)
 body=body.replace('<h5','<h3').replace('</h5>','</h3>')
 body=body.replace('<div class="abstract-title">Abstract</div>','<h2 class="abstract-title">Abstract</h2>')
 references='<section id="references"><h2>References</h2><ol>'+''.join(f'<li id="ref-{html.escape(k)}">{convert(v)}</li>' for k,v in entries)+'</ol></section>'
 head='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="The RoboSWE-ADAPT methods draft: background, task selection, five-trial protocol, harness, verification and proposed evaluation measures."><title>RoboSWE-ADAPT — Draft preprint</title><link rel="stylesheet" href="paper.css"></head><body><a class="skip" href="#paper">Skip to paper</a><nav class="paper-bar" aria-label="Paper navigation"><a class="brand" href="/">RoboSWE <span>ADAPT</span></a><a href="/#results">Benchmark &amp; results ↗</a></nav><main id="paper"><p class="draft-label">DRAFT PREPRINT · METHODS</p><p class="draft-note">Background and methods draft. Evaluation results will be added after the trial series are complete. Not yet submitted to arXiv.</p><div class="paper-links"><a href="source.tex" download>Download TeX source</a><a href="citation.bib" download>Citation (BibTeX)</a><button type="button" id="print-paper">Print / save PDF</button></div>'''
 output=ROOT/'website/paper';output.mkdir(exist_ok=True)
 (output/'index.html').write_text(head+body+references+'<footer><a href="/">Return to RoboSWE-ADAPT</a></footer></main><script src="paper.js" defer></script></body></html>\n')
 if source!=(output/'source.tex').resolve():shutil.copyfile(source,output/'source.tex')
 (output/'source-manifest.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'format':'HTML rendering of the methods draft; no added results','references':len(entries),'converter':'pandoc'},indent=2)+'\n')
 print(json.dumps({'references':len(entries),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
