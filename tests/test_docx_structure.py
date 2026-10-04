"""OOXML checks supplement, never replace, real rendered-page inspection."""
import zipfile
import xml.etree.ElementTree as ET

from autoacoustics import report
from test_report import result


NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}


def test_report_has_valid_zip_chinese_font_repeating_headers_and_bounded_pictures(tmp_path):
    path=report.write_docx(result(),tmp_path/'结构.docx')
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        root=ET.fromstring(archive.read('word/document.xml'))
        style=ET.fromstring(archive.read('word/styles.xml'))
    section=root.find('.//w:sectPr',NS)
    size=section.find('w:pgSz',NS); margins=section.find('w:pgMar',NS)
    width=int(size.attrib['{'+NS['w']+'}w'])
    body=width-int(margins.attrib['{'+NS['w']+'}left'])-int(margins.attrib['{'+NS['w']+'}right'])
    assert (width,int(size.attrib['{'+NS['w']+'}h']))==(12240,15840)
    for extent in root.findall('.//wp:extent',NS):
        assert int(extent.attrib['cx']) <= body * 635  # twips to EMUs
    assert root.findall('.//w:tblHeader',NS)
    assert root.findall('.//w:tblBorders/w:insideH',NS)
    assert any(font.attrib.get('{'+NS['w']+'}eastAsia')=='Microsoft YaHei'
               for font in style.findall('.//w:rFonts',NS))
    for grid in root.findall('.//w:tblGrid',NS):
        assert sum(int(column.attrib['{'+NS['w']+'}w']) for column in grid)<=body+2
    title=next(node for node in style.findall('w:style',NS)
               if node.attrib.get('{'+NS['w']+'}styleId')=='Title')
    assert title.find('w:pPr/w:pBdr',NS) is None


def test_silence_is_explicit_negative_infinity_and_invalid_value_is_empty(tmp_path):
    from dataclasses import replace
    from autoacoustics.model import AnalysisSummary,MetricResult
    sample=replace(result(False),summary=AnalysisSummary({
        'LZeq':MetricResult(float('-inf'),'dB','silent'),
        'Nmean':MetricResult(float('nan'),'sone','ok')}))
    path=report.write_docx(sample,tmp_path/'静音.docx')
    with zipfile.ZipFile(path) as archive:
        raw=archive.read('word/document.xml').decode('utf-8')
    assert '−∞' in raw and '静音' in raw
    assert '指标为无效非有限值' in raw and '>nan<' not in raw
