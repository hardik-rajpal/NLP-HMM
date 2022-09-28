from typing import OrderedDict
import xmltodict
import numpy as np
fname = 'prefs.html'
# fname = 'suffs.html'
def getprefsDict(html:str):
    json_ = xmltodict.parse(html)
    uls = json_['html']['body']['div'][2]['div'][2]['div'][4]['div'][0]['ul']
    return uls
def getItem(li,prefs=True):
    item = ''
    if(type(li)==str):
        item = (li)
    else:
        try:
            item = (li['a']['@title'][:-1]) if prefs else li['a']['@title'][1:]
        except:
            item = (li['a'][0]['@title'][:-1]) if prefs else li['a'][0]['@title'][1:]
    if(item!=''):
        item = item.replace('- (page does not exist','') if prefs else item.replace(' (page does not exist)','')
    return item
with open(fname, "r",encoding='utf-8') as html_file:
    html = html_file.read()
    uls = getprefsDict(html)
    prefs = []
    for ul in uls:
        lis = ul['li']
        for li in lis:
            item = getItem(li,fname.startswith('prefs'))
            prefs.append(item)
    prefs = np.asarray(prefs)
    np.savetxt(f'{fname[:-5]}.txt',prefs,fmt='%s')
