import re
import os
from reportlab.platypus import Table
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import TableStyle

from params import PARAMS_ENV, PARAMS_TRAIN_DDPG, PARAMS_TRAIN_GPS
from params import PARAMS_OBS
from DDPG.params import PARAMS_UTILS, PARAMS_DDPG
from GPS.params import PARAMS_LQG, PARAMS_OFFLINE, PARAMS_ONLINE
# from PPO.params import PARAMS_PPO

PARAMS_OBS = {re.sub(r'\$', '', k): u'\u00B1'+v for k,
              v in PARAMS_OBS.items()}

style = TableStyle([
    ('BACKGROUND', (0, 0), (3, 0), colors.blue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

    ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 14),

    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),

    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
])

ts = TableStyle(
    [
        ('BOX', (0, 0), (-1, -1), 2, colors.black),

        ('LINEBEFORE', (2, 1), (2, -1), 2, colors.red),
        ('LINEABOVE', (0, 2), (-1, 2), 2, colors.green),

        ('GRID', (0, 0), (-1, -1), 2, colors.black),
    ]
)


def drawMyRuler(pdf):
    pdf.drawString(100, 810, 'x100')
    pdf.drawString(200, 810, 'x200')
    pdf.drawString(300, 810, 'x300')
    pdf.drawString(400, 810, 'x400')
    pdf.drawString(500, 810, 'x500')
    pdf.drawString(10, 100, 'y100')
    pdf.drawString(10, 200, 'y200')
    pdf.drawString(10, 300, 'y300')
    pdf.drawString(10, 400, 'y400')
    pdf.drawString(10, 500, 'y500')
    pdf.drawString(10, 600, 'y600')
    pdf.drawString(10, 700, 'y700')
    pdf.drawString(10, 800, 'y800')


def dic_to_list(data):
    lista = [[v, str(k)] for v, k in list(data.items())]
    lista.insert(0, ['Parámetro', 'Valor'])
    return lista


def add_table(pdf, data, x, y):
    data = dic_to_list(data)
    table = Table(data)
    table.setStyle(style)
    table.setStyle(ts)
    table.wrapOn(pdf, 400, 100)
    table.drawOn(pdf, x, y)


def add_text(pdf, textLines, x, y):
    if isinstance(textLines, str):
        textLines = [textLines]
    text = pdf.beginText(x, y)
    text.setFont("Courier", 18)
    text.setFillColor(colors.black)
    for line in textLines:
        text.textLine(line)
    pdf.drawText(text)


def add_image(PATH, pdf, name, x, y, width=500, height=500):
    pdf.drawInlineImage(PATH + name, x, y, width=width,
                        height=height, preserveAspectRatio=True)


def _create_report(PATH):
    fileName = 'Reporte.pdf'
    fileName = PATH + fileName
    documentTitle = 'Document title!'
    title = 'Reporte de Entrenamiento DDPG'
    subTitle = ''
    pdf = canvas.Canvas(fileName)

    pdf.setTitle(documentTitle)
    pdf.drawCentredString(300, 800, title)
    # RGB - Red Green and Blue
    pdf.setFillColorRGB(0, 0, 255)
    pdf.setFont("Courier-Bold", 26)
    pdf.drawCentredString(290, 720, subTitle)
    # drawMyRuler(pdf)
    # add_table(pdf, PARAMS_OBS,100,610)

    # pdf.showPage()

    add_text(pdf, ['Parámetros del', 'Entrenamiento'], 100, 750)
    add_table(pdf, PARAMS_TRAIN_DDPG, 100, 610)

    add_text(pdf, ['Parámetros del',  'Ambiente'], 100, 560)
    add_table(pdf, PARAMS_ENV, 100, 410)

    add_text(pdf, ['Parámetros de', 'DDPG'], 100, 360)
    add_table(pdf, PARAMS_DDPG, 100, 210)

    add_text(pdf, ['Parámetros del', 'Ruido'], 100, 170)
    add_table(pdf, PARAMS_UTILS, 100, 30)

    add_text(pdf, ['Espacio de', 'simulación'], 250, 750)
    add_table(pdf, PARAMS_OBS, 250, 420)

    pdf.showPage()
    add_image(PATH, pdf, 'train_rewards.png', 10, 350, 600, 600)
    add_image(PATH, pdf, 'state_rollouts.png', 30, 10)
    pdf.showPage()
    # Siguiente pagina
    add_image(PATH, pdf, 'action_rollouts.png', 30, 350, 550, 550)
    add_image(PATH, pdf, 'score_rollouts.png', 30, 0, 550, 550)

    # pdf.showPage()
    # add_image(PATH, pdf, 'flight_rollouts.png', 30, 350, 550, 550)
    # add_image(PATH,pdf,'/vuelos_2D.png',30,0,550,550)
    pdf.save()


def create_report(path, title=None, subtitle='', file_name=None,
                  method='ddpg', extra_method='noise'):
    '''
    Genera un documento pdf con el reporte de entrenamiento
    de distintos algoritmos.

    path : str
        Dirección donde será guardado el documento ('Aqui-se-guardara/').
    title : str
        Título que estará en el encabezado de la primera hoja.
    subtitle : str
        Subtítulo que estará en el encabezado de la primera hoja.
    file_name : str
        Nombre del archivo que tendrá el documento ('reporte.pdf').
    method: str
        Nombre del método utilizado en el entrenamiento. Opciones
        disponibles:
        * `ddpg`.
        * `gcl`.
        * `None`

    extra_method: str
        Nombre del método secundario utilizado en el entrenamieto.
        Opciones diponibles:
        * `noise`.
        * `ilqr`.

    '''
    if not isinstance(file_name, str):
        file_name = 'reporte.pdf'
    if not isinstance(title, str):
        title = 'Reporte de entrenamiento'
    file_name = path + file_name

    pdf = canvas.Canvas(file_name)
    pdf.drawCentredString(300, 800, title)
    pdf.setFillColorRGB(0, 0, 255)
    pdf.setFont("Courier-Bold", 26)
    pdf.drawCentredString(290, 760, subtitle)

    add_text(pdf, ['Espacio de', 'observación'], 100, 750)
    add_table(pdf, PARAMS_OBS, 100, 480)

    add_text(pdf, ['Parámetros del', 'ambiente'], 100, 450)
    add_table(pdf, PARAMS_ENV, 100, 250)

    add_text(pdf, ['Parámetros de', 'optimiazación de red'], 100, 220)
    add_table(pdf, PARAMS_DDPG, 100, 50)

    if method == 'ddpg':
        add_text(pdf, ['Parámetros de', 'entrenamiento DDPG'], 350, 750)
        add_table(pdf, PARAMS_TRAIN_DDPG, 350, 600)

    elif method == 'gps':
        add_text(pdf, ['Parámetros de', 'entrenamiento GPS'], 350, 760)
        add_table(pdf, PARAMS_TRAIN_GPS, 350, 600)
    elif method is None:
        pass
    else:
        print(f'La opción method={method} no es válida.')

    if extra_method == 'noise':
        add_text(pdf, ['Parámetros de', 'ruido'], 350, 550)
        add_table(pdf, PARAMS_UTILS, 400, 340)

    elif extra_method == 'ilqr':
        add_text(pdf, ['Parámetros de', 'iLQR'], 350, 580)
        add_table(pdf, PARAMS_LQG, 350, 400)

        add_text(pdf, ['Parámetros de', '"Offline control"'], 350, 390)
        add_table(pdf, PARAMS_OFFLINE, 350, 190)

        add_text(pdf, ['Parámetros de', '"Online control"'], 350, 170)
        add_table(pdf, PARAMS_ONLINE, 350, 60)

    elif method is None:
        pass
    else:
        print(f'La opción method={extra_method} no es válida.')

    pdf.showPage()
    if os.path.exists(path + 'train_performance.png'):
        add_text(pdf, ['Rendimiento de entrenamiento'], 30, 750)
        add_image(path, pdf, 'train_performance.png', 100, 400, 350, 350)
    add_text(pdf, ['Simulaciones (estados)'], 30, 390)
    add_image(path, pdf, 'state_rollouts.png', 30, -10, 500, 500)

    pdf.showPage()
    add_text(pdf, ['Simulaciones (acciones)'], 30, 750)
    add_image(path, pdf, 'action_rollouts.png', 30, 350, 500, 500)
    add_text(pdf, ['Simulaciones (penalizaciones)'], 30, 400)
    add_image(path, pdf, 'score_rollouts.png', 30, 0, 500, 500)

    pdf.save()


if __name__ == '__main__':
    create_report('results_gcl/22_12_08_18_55/',
                  title='Reporte de entrenamiento GCL',
                  method='gcl', extra_method='ilqr')
