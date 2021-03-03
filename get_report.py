from DDPG.params import PARAMS_UTILS, PARAMS_DDPG
from reportlab.platypus import Table
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import TableStyle




fileName = 'MyDoc.pdf'
documentTitle = 'Document title!'
title = 'Tasmanian devil'
subTitle = 'The largest carnivorous marsupial'
pdf = canvas.Canvas(fileName)
pdf.setTitle(documentTitle)



style = TableStyle([
    ('BACKGROUND', (0,0), (3,0), colors.blue),
    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),

    ('ALIGN',(0,0),(-1,-1),'CENTER'),

    ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 14),

    ('BOTTOMPADDING', (0,0), (-1,0), 12),

    ('BACKGROUND',(0,1),(-1,-1),colors.beige),
])

ts = TableStyle(
    [
    ('BOX',(0,0),(-1,-1),2,colors.black),

    ('LINEBEFORE',(2,1),(2,-1),2,colors.red),
    ('LINEABOVE',(0,2),(-1,2),2,colors.green),

    ('GRID',(0,0),(-1,-1),2,colors.black),
    ]
)


def drawMyRuler(pdf):
    pdf.drawString(100,810, 'x100')
    pdf.drawString(200,810, 'x200')
    pdf.drawString(300,810, 'x300')
    pdf.drawString(400,810, 'x400')
    pdf.drawString(500,810, 'x500')

    pdf.drawString(10,100, 'y100')
    pdf.drawString(10,200, 'y200')
    pdf.drawString(10,300, 'y300')
    pdf.drawString(10,400, 'y400')
    pdf.drawString(10,500, 'y500')
    pdf.drawString(10,600, 'y600')
    pdf.drawString(10,700, 'y700')
    pdf.drawString(10,800, 'y800')

drawMyRuler(pdf)


textLines = [
'The Tasmanian devil (Sarcophilus harrisii) is',
'a carnivorous marsupial of the family',
'Dasyuridae.'
]

 

text = pdf.beginText(40, 680)
text.setFont("Courier", 18)
text.setFillColor(colors.red)
for line in textLines:
    text.textLine(line)

pdf.drawText(text)

def dic_to_list(data):
    lista = [[v, str(k)] for v, k in list(data.items())]
    lista.insert(0, ['Param', 'Valor'])
    return lista

def add_table(data,x,y):
    data = dic_to_list(data)
    table = Table(data)
    table.setStyle(style)
    table.setStyle(ts)
    table.wrapOn(pdf,400,100)
    table.drawOn(pdf, x, y)

# pdf.drawInlineImage(image, 130, 400)
add_table(PARAMS_DDPG,200,600)
add_table(PARAMS_UTILS,200,550)

pdf.save()