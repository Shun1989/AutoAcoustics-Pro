"""Compose preserved pages for review; never marks them as read."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('document')
    parser.add_argument('first',type=int)
    parser.add_argument('last',type=int)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'output/knowledge_corpus/source_manifest.json').read_text(encoding='utf-8'))
    matches=[d for d in manifest['documents'] if d['relative_path']==args.document or args.document in d.get('aliases',[])]
    if len(matches)!=1:raise ValueError('Document must identify one preserved source')
    document=matches[0]
    if not 1<=args.first<=args.last<=document['page_count']:raise ValueError('Page range invalid')
    output=ROOT/'output/knowledge_corpus/review_contact_sheets'/document['sha256'][:12]
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for start in range(args.first,args.last+1,4):
        selected=document['pages'][start-1:min(start+3,args.last)]
        pages=[Image.open(p['render_path']).convert('RGB') for p in selected]
        width=max(p.width for p in pages);height=max(p.height for p in pages)
        sheet=Image.new('RGB',(width*2,(height+30)*2),'#e6e6e6')
        draw=ImageDraw.Draw(sheet)
        for index,(page,meta) in enumerate(zip(pages,selected)):
            x=(index%2)*width;y=(index//2)*(height+30)
            draw.text((x+5,y+5),f'PDF physical page {meta["page"]}',fill='black')
            sheet.paste(page,(x,y+30))
        path=output/f'{selected[0]["page"]:04d}-{selected[-1]["page"]:04d}.png'
        sheet.save(path)
        records.append({'pages':[p['page'] for p in selected],'path':str(path),'source_sha256':document['sha256']})
    print(json.dumps(records,ensure_ascii=False))

if __name__=='__main__':main()
